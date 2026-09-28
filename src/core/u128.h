// Minimal unsigned 128-bit integer for overflow-checked host arithmetic. MSVC has no __int128,
// and the project uses one code path on every platform; only the operations actually needed by
// the context-cost and prefill-work calculations are provided.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer {

struct U128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    constexpr U128() noexcept = default;
    // Implicit, mirroring the conversion semantics of unsigned __int128 at the call sites.
    constexpr U128(std::uint64_t value) noexcept : lo(value), hi(0) {}
    constexpr U128(std::uint64_t low, std::uint64_t high) noexcept : lo(low), hi(high) {}

    // Full 128-bit product of two 64-bit values (portable limb decomposition).
    [[nodiscard]] static constexpr U128 multiply(std::uint64_t a, std::uint64_t b) noexcept {
        constexpr std::uint64_t kLow32 = 0xFFFFFFFFu;
        const std::uint64_t a_lo       = a & kLow32;
        const std::uint64_t a_hi       = a >> 32;
        const std::uint64_t b_lo       = b & kLow32;
        const std::uint64_t b_hi       = b >> 32;

        const std::uint64_t p0 = a_lo * b_lo;
        const std::uint64_t p1 = a_lo * b_hi;
        const std::uint64_t p2 = a_hi * b_lo;
        const std::uint64_t p3 = a_hi * b_hi;

        const std::uint64_t middle = (p0 >> 32) + (p1 & kLow32) + (p2 & kLow32);
        return U128{((middle & kLow32) << 32) | (p0 & kLow32),
                    p3 + (p1 >> 32) + (p2 >> 32) + (middle >> 32)};
    }

    [[nodiscard]] static constexpr U128 max_value() noexcept {
        return U128{std::numeric_limits<std::uint64_t>::max(),
                    std::numeric_limits<std::uint64_t>::max()};
    }

    // Multiplication is only defined for 64-bit operands; wider values would silently lose bits.
    [[nodiscard]] friend constexpr U128 operator*(U128 left, U128 right) {
        if (left.hi != 0 || right.hi != 0) { throw std::overflow_error("U128 operand exceeds 64 bits"); }
        return multiply(left.lo, right.lo);
    }

    [[nodiscard]] friend constexpr U128 operator+(U128 left, U128 right) noexcept {
        const std::uint64_t lo = left.lo + right.lo;
        const std::uint64_t carry = lo < left.lo ? 1u : 0u;
        return U128{lo, left.hi + right.hi + carry};
    }

    [[nodiscard]] friend constexpr U128 operator-(U128 left, U128 right) noexcept {
        const std::uint64_t lo = left.lo - right.lo;
        const std::uint64_t borrow = left.lo < right.lo ? 1u : 0u;
        return U128{lo, left.hi - right.hi - borrow};
    }

    [[nodiscard]] friend constexpr bool operator<(U128 left, U128 right) noexcept {
        return left.hi != right.hi ? left.hi < right.hi : left.lo < right.lo;
    }

    [[nodiscard]] friend constexpr bool operator>(U128 left, U128 right) noexcept {
        return right < left;
    }

    [[nodiscard]] friend constexpr bool operator<=(U128 left, U128 right) noexcept {
        return !(right < left);
    }

    [[nodiscard]] friend constexpr bool operator>=(U128 left, U128 right) noexcept {
        return !(left < right);
    }

    // Logical shifts; amounts of 128 or more yield zero.
    [[nodiscard]] friend constexpr U128 operator<<(U128 value, unsigned int amount) noexcept {
        if (amount == 0) { return value; }
        if (amount >= 128) { return U128{}; }
        if (amount >= 64) { return U128{0, value.lo << (amount - 64)}; }
        return U128{value.lo << amount, (value.hi << amount) | (value.lo >> (64 - amount))};
    }

    [[nodiscard]] friend constexpr U128 operator>>(U128 value, unsigned int amount) noexcept {
        if (amount == 0) { return value; }
        if (amount >= 128) { return U128{}; }
        if (amount >= 64) { return U128{value.hi >> (amount - 64), 0}; }
        return U128{(value.lo >> amount) | (value.hi << (64 - amount)), value.hi >> amount};
    }

    // Truncating conversion to the low 64 bits.
    explicit constexpr operator std::uint64_t() const noexcept { return lo; }
};

} // namespace ninfer
