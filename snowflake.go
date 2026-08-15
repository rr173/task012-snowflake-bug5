package main

import (
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// Bit layout of a 64-bit ID (high bits first):
//
//	1 bit  sign        (always 0)
//	41 bit timestamp   (ms since the custom epoch)
//	10 bit machineID
//	12 bit sequence    (per millisecond)
const (
	signBit        = 63
	timestampBits  = 41
	machineBits    = 10
	sequenceBits   = 12
	machineShift   = sequenceBits                     // 12
	timestampShift = machineBits + sequenceBits       // 22
	maxMachineID   = int64((1 << machineBits) - 1)    // 1023
	maxSequence    = uint64((1 << sequenceBits) - 1)  // 4095
	maxTimestamp   = uint64((1 << timestampBits) - 1) // 2^41 - 1
	sequenceMask   = uint64((1 << sequenceBits) - 1)  // 0xFFF
	machineMask    = uint64((1 << machineBits) - 1)   // 0x3FF

	// EpochMs is the custom epoch (2024-01-01T00:00:00Z) in Unix milliseconds.
	// All timestamps stored in an ID are measured from this instant.
	EpochMs int64 = 1704067200000

	// MaxBatch is the largest number of IDs a single request may generate.
	MaxBatch = 5000

	// waitBudget bounds how long next() spins waiting for the next millisecond
	// when the sequence overflows in the current one. With a live clock this is
	// reached only under extreme burst load; with a frozen test clock it makes
	// sequence exhaustion observable instead of looping forever.
	waitBudget = 5 * time.Millisecond
)

// compose assembles a 64-bit ID from its three fields. It assumes the caller has
// already validated the ranges.
func compose(ts uint64, machineID int64, seq uint64) uint64 {
	return (ts << timestampShift) | (uint64(machineID) << machineShift) | seq
}

// decompose splits a 64-bit ID into its timestamp, machineID and sequence.
func decompose(id uint64) (ts uint64, machineID int64, seq uint64) {
	ts = id >> timestampShift
	machineID = int64((id >> machineShift) & machineMask)
	seq = id & sequenceMask
	return
}

// ---- structured errors carrying HTTP semantics ----

type statusErr struct {
	code int
	msg  string
}

func (e *statusErr) Error() string { return e.msg }

func badRequest(format string, a ...any) error {
	return &statusErr{code: http.StatusBadRequest, msg: fmt.Sprintf(format, a...)}
}
func conflict(format string, a ...any) error {
	return &statusErr{code: http.StatusConflict, msg: fmt.Sprintf(format, a...)}
}
func notFound(format string, a ...any) error {
	return &statusErr{code: http.StatusNotFound, msg: fmt.Sprintf(format, a...)}
}
func serviceUnavailable(format string, a ...any) error {
	return &statusErr{code: http.StatusServiceUnavailable, msg: fmt.Sprintf(format, a...)}
}

// clockBackward and sequenceExhausted are the two transient failures that make a
// snowflake generator safe. They are returned as 503 so callers know to retry.
func clockBackward(now, last int64) error {
	return serviceUnavailable("clock moved backward (now=%d < last=%d): refusing to generate a duplicate ID", now, last)
}
func sequenceExhausted() error {
	return serviceUnavailable("sequence exhausted in the current millisecond; retry later")
}

// ---- per-machine generator state ----

// generator holds the monotonic state for one registered machine. Its fields are
// guarded by the owning Service's mutex.
type generator struct {
	machineID int64
	lastTs    int64  // last absolute millisecond at which an ID was produced
	seq       uint64 // last sequence value used within lastTs
	clock     func() int64
}

// next produces one ID. On clock rollback it refuses; on sequence overflow it
// waits (up to waitBudget) for the next millisecond and only errors if the clock
// cannot advance.
func (g *generator) next() (uint64, error) {
	now := g.clock()
	if now < g.lastTs {
		return 0, clockBackward(now, g.lastTs)
	}
	if now == g.lastTs {
		g.seq = (g.seq + 1) & sequenceMask
		if g.seq == 0 {
			// Used all 4096 slots in this millisecond; try to advance.
			advanced := g.waitNextMs(g.lastTs)
			if advanced <= g.lastTs {
				return 0, sequenceExhausted()
			}
			g.lastTs = advanced
			// seq stays 0 (first ID of the new millisecond).
		}
	} else {
		g.lastTs = now
		g.seq = 0
	}
	// The ID stores the millisecond offset from the custom epoch, not the
	// absolute Unix time, so that decompose yields (genTime - EpochMs).
	// g.lastTs-EpochMs is non-negative here: the guard below rejects a clock
	// before the epoch before this value can be used.
	ts := uint64(g.lastTs - EpochMs)
	if int64(g.lastTs-EpochMs) < 0 {
		return 0, serviceUnavailable("clock is before the custom epoch")
	}
	if ts > maxTimestamp {
		return 0, serviceUnavailable("timestamp overflow: epoch distance exceeds %d bits", timestampBits)
	}
	return compose(ts, g.machineID, g.seq), nil
}

// waitNextMs polls the clock until it observes a value strictly greater than
// `from`, or until waitBudget of real time elapses. It returns the advanced
// value, or `from` if the clock could not be advanced.
func (g *generator) waitNextMs(from int64) int64 {
	deadline := time.Now().Add(waitBudget)
	for {
		n := g.clock()
		if n > from {
			return n
		}
		if time.Now().After(deadline) {
			return from
		}
		time.Sleep(50 * time.Microsecond)
	}
}

// ---- machine registry / service ----

// Machine is the public view of a registered generator.
type Machine struct {
	MachineID    int64 `json:"machineID"`
	RegisteredAt int64 `json:"registeredAt"` // monotonic registration sequence (1-based)
}

// Inspection is the parsed view of an existing ID.
type Inspection struct {
	ID          string `json:"id"`
	Timestamp   uint64 `json:"timestamp"`
	MachineID   int64  `json:"machineID"`
	Sequence    uint64 `json:"sequence"`
	GeneratedAt string `json:"generatedAt"` // RFC3339 of epoch + timestamp
}

// Service is an in-memory snowflake registry. It is safe for concurrent use.
type Service struct {
	mu       sync.Mutex
	machines map[int64]*generator
	order    []int64 // machineIDs in registration order
	seq      int64   // registration sequence
	clock    func() int64
}

// NewService returns a service backed by the real wall clock.
func NewService() *Service {
	return newServiceWithClock(realClock)
}

// newServiceWithClock returns a service whose generators read time from `clock`.
// It exists for tests that need deterministic control over the clock.
func newServiceWithClock(clock func() int64) *Service {
	if clock == nil {
		clock = realClock
	}
	return &Service{machines: map[int64]*generator{}, clock: clock}
}

func realClock() int64 { return time.Now().UnixMilli() }

// RegisterMachine adds a machine with the given ID. The ID must be in
// [0, maxMachineID] and not already registered.
func (s *Service) RegisterMachine(machineID int64) (*Machine, error) {
	if machineID < 0 || machineID > maxMachineID {
		return nil, badRequest("machineID must be in [0, %d], got %d", maxMachineID, machineID)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.machines[machineID]; exists {
		return nil, conflict("machine %d already registered", machineID)
	}
	s.seq++
	s.machines[machineID] = &generator{machineID: machineID, clock: s.clock}
	s.order = append(s.order, machineID)
	return &Machine{MachineID: machineID, RegisteredAt: s.seq}, nil
}

// Generate produces `count` IDs on the named machine.
func (s *Service) Generate(machineID int64, count int) ([]uint64, error) {
	if count < 1 {
		return nil, badRequest("count must be >= 1, got %d", count)
	}
	if count > MaxBatch {
		return nil, badRequest("count must be <= %d, got %d", MaxBatch, count)
	}
	s.mu.Lock()
	g, ok := s.machines[machineID]
	s.mu.Unlock()
	if !ok {
		return nil, notFound("machine %d not registered", machineID)
	}
	ids := make([]uint64, count)
	for i := 0; i < count; i++ {
		id, err := g.next()
		if err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, nil
}

// Inspect parses an ID string and returns its decoded fields.
func (s *Service) Inspect(idStr string) (*Inspection, error) {
	id, err := strconv.ParseUint(idStr, 10, 64)
	if err != nil {
		return nil, badRequest("id is not a valid 64-bit unsigned integer: %v", err)
	}
	if id>>signBit == 0 {
		return nil, badRequest("id sign bit must be 0")
	}
	ts, machineID, seq := decompose(id)
	return &Inspection{
		ID:          strconv.FormatUint(id, 10),
		Timestamp:   ts,
		MachineID:   machineID,
		Sequence:    seq,
		GeneratedAt: time.UnixMilli(EpochMs + int64(ts)).UTC().Format(time.RFC3339Nano),
	}, nil
}

// GetMachine returns one machine by ID.
func (s *Service) GetMachine(machineID int64) (*Machine, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.machines[machineID]; !ok {
		return nil, notFound("machine %d not registered", machineID)
	}
	// RegisteredAt is reconstructed from the order slice.
	for i, id := range s.order {
		if id == machineID {
			return &Machine{MachineID: machineID, RegisteredAt: int64(i + 1)}, nil
		}
	}
	return nil, notFound("machine %d not registered", machineID)
}

// ListMachines returns machines in registration order.
func (s *Service) ListMachines() []*Machine {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*Machine, 0, len(s.order))
	for i, id := range s.order {
		out = append(out, &Machine{MachineID: id, RegisteredAt: int64(i + 1)})
	}
	return out
}

// RemoveMachine deletes a machine. Generating IDs from it afterwards fails.
func (s *Service) RemoveMachine(machineID int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.machines[machineID]; !ok {
		return notFound("machine %d not registered", machineID)
	}
	delete(s.machines, machineID)
	return nil
}
