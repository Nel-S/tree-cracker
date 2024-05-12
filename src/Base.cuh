#pragma once

#include "nonCudaSupport.cuh"
#include <algorithm>
#include <bit>

// A two-dimensional coordinate.
struct Coordinate {
	int32_t x, z;
	__host__ __device__ constexpr Coordinate() noexcept : x(), z() {}
	__host__ __device__ constexpr Coordinate(const int32_t x, const int32_t z) noexcept : x(x), z(z) {}
};

// An inclusive range of 32-bit integers.
// The range can be nullified (meaning contains() always returns true) by setting lowerBound to -1.
struct IntInclusiveRange {
	int32_t lowerBound, upperBound;

	__device__ constexpr IntInclusiveRange() noexcept : lowerBound(INT32_MIN), upperBound(INT32_MAX) {}
	__device__ constexpr IntInclusiveRange(const IntInclusiveRange &other) noexcept : lowerBound(other.lowerBound), upperBound(other.upperBound) {}
	__device__ constexpr IntInclusiveRange(int32_t value) noexcept : lowerBound(value), upperBound(value) {}
	__device__ constexpr IntInclusiveRange(int32_t lowerBound, int32_t upperBound) noexcept : lowerBound(std::min(lowerBound, upperBound)), upperBound(std::max(lowerBound, upperBound)) {}
	// Initialize based on the intersection of two ranges.
	__device__ constexpr IntInclusiveRange(const IntInclusiveRange &range1, const IntInclusiveRange &range2) noexcept : lowerBound(std::max(range1.lowerBound, range2.lowerBound)), upperBound(std::min(range1.upperBound, range2.upperBound)) {}

	// Returns if a value falls within the range.
	__device__ constexpr bool contains(int32_t value) const noexcept {
		return this->lowerBound <= value && value <= this->upperBound;
	}

	// Returns the range's range.
	__device__ constexpr int32_t getRange() const noexcept {
		return this->upperBound - this->lowerBound + 1;
	}

	__host__ void printRangeComparison(const IntInclusiveRange &originalRange, const IntInclusiveRange &allowedBounds) const noexcept {	
		if (SILENT_MODE) return;
		if (!this->contains(originalRange.lowerBound)) std::fprintf(stderr, "WARNING: Input's lower bound (%" PRId32 ") fell outside of allowed range [%" PRId32 ", %" PRId32 "] and so was replaced with %" PRId32 ".\n", originalRange.lowerBound, allowedBounds.lowerBound, allowedBounds.upperBound, this->lowerBound);
		if (!this->contains(originalRange.upperBound)) std::fprintf(stderr, "WARNING: Input's upper bound (%" PRId32 ") fell outside of allowed range [%" PRId32 ", %" PRId32 "] and so was replaced with %" PRId32 ".\n", originalRange.upperBound, allowedBounds.lowerBound, allowedBounds.upperBound, this->upperBound);
	
	}
};
typedef IntInclusiveRange PossibleHeightsRange;
typedef IntInclusiveRange PossibleRadiiRange;



// Returns 2**bits.
constexpr uint64_t twoToThePowerOf(const int32_t bits) noexcept {
	return UINT64_C(1) << bits;
}

// Returns a [bits]-bit-wide mask.
constexpr uint64_t getBitmask(const int32_t bits) noexcept {
	return twoToThePowerOf(bits) - UINT64_C(1);
}

// Returns the lowest [bits] bits of value.
constexpr uint64_t mask(const uint64_t value, const int32_t bits) noexcept {
	return value & getBitmask(bits);
}

// Returns x such that value * x == 1 (mod 2^64).
constexpr uint64_t inverseModulo(const uint64_t value) noexcept {
	uint64_t x = ((value << 1 ^ value) & 4) << 1 ^ value;
	x += x - value * x * x;
	x += x - value * x * x;
	x += x - value * x * x;
	return x + (x - value * x * x);
}

// Returns the number of trailing zeroes in value.
constexpr uint32_t getNumberOfTrailingZeroes(const uint64_t value) noexcept {
	return std::__countr_zero(value);
}

// Returns the number of leading zeroes in value.
constexpr uint32_t getNumberOfLeadingZeroes(const uint64_t value) noexcept {
	return std::__countl_zero(value);
}

// Returns the compile-time ceiling of a number.
// From s3cur3 on Stack Overflow (https://stackoverflow.com/a/66146159).
constexpr int64_t constexprCeil(const double x) {
	int64_t xAsInteger = static_cast<int64_t>(x);
    return xAsInteger + static_cast<int64_t>(x > xAsInteger);
}

// Returns a compile-time approximation of e^x. (This becomes less accurate the further one drifts from 0.)
constexpr double constexprExp(const double x) {
	return 1. + x*(1. + x/2.*(1. + x/3.*(1. + x/4.*(1. + x/5.*(1. + x/6.*(1. + x/7.))))));
}

// Returns a compile-time approximation of log(x). (This becomes less accurate the further one drifts from 0.)
constexpr double constexprLog(const double x) {
	double y = x;
	for (uint32_t i = 0; i < 15; ++i) {
		double yExp = constexprExp(y);
		y += 2*(x - yExp)/(x + yExp);
	}
	return y;
}

// Returns a compile-time approximation of log_2(x). (This becomes less accurate the further one drifts from 0.)
constexpr double constexprLog2(const double x) {
	return constexprLog(x)/0.693147180559945309417; // ln(2)
}