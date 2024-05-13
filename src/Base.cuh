#ifndef __BASE_CUH
#define __BASE_CUH

#include "nonCudaSupport.cuh"

// Returns the compile-time minimum of a and b.
__host__ __device__ constexpr int32_t constexprMin(const int32_t a, const int32_t b) {
	return a < b ? a : b;
}
// Returns the compile-time minimum of a and b.
__host__ __device__ constexpr uint32_t constexprMin(const uint32_t a, const uint32_t b) {
	return a < b ? a : b;
}
// Returns the compile-time minimum of a and b.
__host__ __device__ constexpr int64_t constexprMin(const int64_t a, const int64_t b) {
	return a < b ? a : b;
}
// Returns the compile-time minimum of a and b.
__host__ __device__ constexpr uint64_t constexprMin(const uint64_t a, const uint64_t b) {
	return a < b ? a : b;
}

// Returns the compile-time maximum of a and b.
__host__ __device__ constexpr int32_t constexprMax(const int32_t a, const int32_t b) {
	return a > b ? a : b;
}
__host__ __device__ constexpr uint32_t constexprMax(const uint32_t a, const uint32_t b) {
	return a > b ? a : b;
}
// Returns the compile-time maximum of a and b.
__host__ __device__ constexpr int64_t constexprMax(const int64_t a, const int64_t b) {
	return a > b ? a : b;
}
__host__ __device__ constexpr uint64_t constexprMax(const uint64_t a, const uint64_t b) {
	return a > b ? a : b;
}

// Returns 2**bits.
__host__ __device__ constexpr uint64_t twoToThePowerOf(const int32_t bits) noexcept {
	return UINT64_C(1) << bits;
}

// Returns a [bits]-bit-wide mask.
__host__ __device__ constexpr uint64_t getBitmask(const int32_t bits) noexcept {
	return twoToThePowerOf(bits) - UINT64_C(1);
}

// Returns the lowest [bits] bits of value.
__host__ __device__ constexpr uint64_t mask(const uint64_t value, const int32_t bits) noexcept {
	return value & getBitmask(bits);
}

// Returns x such that value * x == 1 (mod 2^64).
__host__ __device__ constexpr uint64_t inverseModulo(const uint64_t value) noexcept {
	uint64_t x = ((value << 1 ^ value) & 4) << 1 ^ value;
	x += x - value * x * x;
	x += x - value * x * x;
	x += x - value * x * x;
	return x + (x - value * x * x);
}

// Returns the number of trailing zeroes in value.
__host__ __device__ constexpr uint32_t getNumberOfTrailingZeroes(const uint64_t value) noexcept {
	if (!value) return 64;
	uint32_t count = 0;
	for (uint64_t v = value; !(v & 1); v >>= 1) ++count;
	return count;
}

// Returns the number of leading zeroes in value.
__host__ __device__ constexpr uint32_t getNumberOfLeadingZeroes(const uint64_t value) noexcept {
	if (!value) return 64;
	uint32_t count = 0;
	for (uint64_t v = value; !(v & 0x8000000000000000); v <<= 1) ++count;
	return count;
}

__host__ __device__ constexpr uint32_t getNumberOfOnesIn(uint32_t x) {
	uint32_t count = 0;
	for (; static_cast<bool>(x); x >>= 1) count += static_cast<int>(x & 1);
	return count;
}

// Returns the compile-time ceiling of a number.
// From s3cur3 on Stack Overflow (https://stackoverflow.com/a/66146159).
__host__ __device__ constexpr int64_t constexprCeil(const double x) {
	int64_t xAsInteger = static_cast<int64_t>(x);
    return xAsInteger + static_cast<int64_t>(x > xAsInteger);
}

// Returns a compile-time approximation of e^x. (This becomes less accurate the further one drifts from 0.)
__host__ __device__ constexpr double constexprExp(const double x) {
	return 1. + x*(1. + x/2.*(1. + x/3.*(1. + x/4.*(1. + x/5.*(1. + x/6.*(1. + x/7.))))));
}

// Returns a compile-time approximation of log(x). (This becomes less accurate the further one drifts from 0.)
__host__ __device__ constexpr double constexprLog(const double x) {
	double y = x;
	for (uint32_t i = 0; i < 15; ++i) {
		double yExp = constexprExp(y);
		y += 2*(x - yExp)/(x + yExp);
	}
	return y;
}

// Returns a compile-time approximation of log_2(x). (This becomes less accurate the further one drifts from 0.)
__host__ __device__ constexpr double constexprLog2(const double x) {
	return constexprLog(x)/0.693147180559945309417; // ln(2)
}


// A two-dimensional coordinate.
struct Coordinate {
	int32_t x, z;
	__host__ __device__ constexpr Coordinate() noexcept : x(), z() {}
	__host__ __device__ constexpr Coordinate(const int32_t x, const int32_t z) noexcept : x(x), z(z) {}
};

// An inclusive range of 32-bit integers.
struct IntInclusiveRange {
	int32_t lowerBound, upperBound;
	static constexpr int32_t NO_MINIMUM = INT32_MIN;
	static constexpr int32_t NO_MAXIMUM = INT32_MAX;

	__device__ constexpr IntInclusiveRange() noexcept : lowerBound(this->NO_MINIMUM), upperBound(this->NO_MAXIMUM) {}
	__device__ constexpr IntInclusiveRange(const IntInclusiveRange &other) noexcept : lowerBound(other.lowerBound), upperBound(other.upperBound) {}
	__device__ constexpr IntInclusiveRange(int32_t value) noexcept : lowerBound(value), upperBound(value) {}
	__device__ constexpr IntInclusiveRange(int32_t lowerBound, int32_t upperBound) noexcept : lowerBound(constexprMin(lowerBound, upperBound)), upperBound(constexprMax(lowerBound, upperBound)) {}
	// Initialize based on the intersection of two ranges.
	__device__ constexpr IntInclusiveRange(const IntInclusiveRange &range1, const IntInclusiveRange &range2) noexcept : lowerBound(constexprMax(range1.lowerBound, range2.lowerBound)), upperBound(constexprMin(range1.upperBound, range2.upperBound)) {}

	// Returns if a value falls within the range.
	__device__ constexpr bool contains(int32_t value) const noexcept {
		return this->lowerBound <= value && value <= this->upperBound;
	}

	// Returns the range's range.
	__host__ __device__ constexpr int32_t getRange() const noexcept {
		return this->upperBound - this->lowerBound + 1;
	}
};
typedef IntInclusiveRange PossibleHeightsRange;
typedef IntInclusiveRange PossibleRadiiRange;

#endif