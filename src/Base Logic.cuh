#ifndef __BASE_CUH
#define __BASE_CUH

#include "Non-CUDA Support.cuh"
#include <string>

// Returns the compile-time minimum of two comparable values.
template <class T>
__host__ __device__ [[nodiscard]] constexpr T constexprMin(const T &a, const T &b) noexcept {
	return a < b ? a : b;
}
// Returns the compile-time maximum of two comparable values.
template <class T>
__host__ __device__ [[nodiscard]] constexpr T constexprMax(const T &a, const T &b) noexcept {
	return a > b ? a : b;
}
// Swaps two elements in compilation-time.
// TODO: Create backup for std::move in case it isn't constexpr on a device?
template <class T>
__device__ constexpr void constexprSwap(T &first, T &second) noexcept {
	auto __temp = std::move(first);
	first = std::move(second);
	second = std::move(__temp);
}

// Returns the compile-time floor of a real number.
__host__ __device__ [[nodiscard]] constexpr int64_t constexprFloor(const double x) noexcept {
	int64_t xAsInteger = static_cast<int64_t>(x);
    return xAsInteger - static_cast<int64_t>(x < xAsInteger);
}

/* Returns the compile-time ceiling of a real number.
   From s3cur3 on Stack Overflow (https://stackoverflow.com/a/66146159).*/
__host__ __device__ [[nodiscard]] constexpr int64_t constexprCeil(const double x) noexcept {
	int64_t xAsInteger = static_cast<int64_t>(x);
    return xAsInteger + static_cast<int64_t>(x > xAsInteger);
}

/* Returns the compile-time rounded value of a real number.
   From Verdy p on Wikipedia (https://en.wikipedia.org/w/index.php?diff=378485717).*/
__host__ __device__ [[nodiscard]] constexpr int64_t constexprRound(const double x) noexcept {
	return constexprFloor(x + 0.5);
}
/* Returns the compile-time factorial of a natural number.
   WARNING: this will overflow (and thus give results modulo 2^64 instead) for any value greater than 20.*/
__host__ __device__ [[nodiscard]] constexpr uint64_t constexprFactorial(const uint64_t n) noexcept {
	uint64_t out = 1;
	for (uint64_t i = 2; i <= n; ++i) out *= i;
	return out;
}

/* Returns the number of ways to choose k objects from a set of n objects, where the order the selections are made does not matter.
   WARNING: this will overflow (and thus give results modulo 2^64 instead) if n > 40 and k > 20.*/
// TODO: Tighten bound in description
__host__ __device__ [[nodiscard]] constexpr uint64_t constexprCombination(const uint64_t n, const uint64_t k) noexcept {
	uint64_t out = 1;
	for (uint64_t i = n; constexprMax(k, n - k) < i; --i) out *= i;
	return out/constexprFactorial(constexprMin(k, n - k));
}

/* Returns a compile-time *approximation* of e^x. (I.e. this becomes less accurate the further one drifts from 0.)
   Derived from Nayuki on Wikipedia (https://en.wikipedia.org/w/index.php?diff=2860008).*/
__host__ __device__ [[nodiscard]] constexpr double constexprExp(const double x) noexcept {
	double approximation = 1.;
	for (uint64_t i = 25; i; --i) approximation = 1. + x*approximation/i;
	return approximation;
}

/* Returns a compile-time *approximation* of ln(x). (I.e. this becomes less accurate the further one drifts from 0.)
   From JRSpriggs on Wikipedia (https://en.wikipedia.org/w/index.php?diff=592947838).*/
__host__ __device__ [[nodiscard]] constexpr double constexprLog(const double x) noexcept {
	double approximation = x;
	for (uint32_t i = 0; i < 25; ++i) {
		double approximationExponent = constexprExp(approximation);
		approximation += 2*(x - approximationExponent)/(x + approximationExponent);
	}
	return approximation;
}

/* Returns a compile-time *approximation* of log_2(x). (I.e. this becomes less accurate the further one drifts from 0.)
   From David Eppstein on Wikipedia (https://en.wikipedia.org/w/index.php?diff=629917843).*/
__host__ __device__ [[nodiscard]] constexpr double constexprLog2(const double x) noexcept {
	return constexprLog(x)/0.693147180559945309417; // ln(2)
}


// Returns 2**bits.
__host__ __device__ [[nodiscard]] constexpr uint64_t twoToThePowerOf(const int32_t bits) noexcept {
	return UINT64_C(1) << bits;
}

// Returns a [bits]-bit-wide mask.
__host__ __device__  [[nodiscard]] constexpr uint64_t getBitmask(const int32_t bits) noexcept {
	return twoToThePowerOf(bits) - UINT64_C(1);
}

// Returns the lowest [bits] bits of value.
__host__ __device__  [[nodiscard]] constexpr uint64_t mask(const uint64_t value, const int32_t bits) noexcept {
	return value & getBitmask(bits);
}

// Returns x such that value * x == 1 (mod 2^64).
__host__ __device__  [[nodiscard]] constexpr uint64_t inverseModulo(const uint64_t value) noexcept {
	uint64_t x = ((value << 1 ^ value) & 4) << 1 ^ value;
	x += x - value * x * x;
	x += x - value * x * x;
	x += x - value * x * x;
	return x + (x - value * x * x);
}

// Returns the number of trailing zeroes in value.
__host__ __device__  [[nodiscard]] constexpr uint32_t getNumberOfTrailingZeroes(const uint64_t value) noexcept {
	if (!value) return 64;
	uint32_t count = 0;
	for (uint64_t v = value; !(v & 1); v >>= 1) ++count;
	return count;
}

// Returns the number of leading zeroes in value.
__host__ __device__  [[nodiscard]] constexpr uint32_t getNumberOfLeadingZeroes(const uint64_t value) noexcept {
	if (!value) return 64;
	uint32_t count = 0;
	for (uint64_t v = value; !(v & (UINT64_C(1) << 63)); v <<= 1) ++count;
	return count;
}

__host__ __device__ [[nodiscard]] constexpr uint32_t getNumberOfOnesIn(const uint32_t x) noexcept {
	uint32_t count = 0;
	for (uint32_t i = x; static_cast<bool>(i); i >>= 1) count += static_cast<int>(i & 1);
	return count;
}


// For pre-C++17
// TODO: Also create (working) const char * implementation?
std::string getFilepathStem(const std::string &filepath) {
	return filepath.substr(0, filepath.rfind('.'));
}

// For pre-C++17
const char *getFilepathExtension(const char *filepath) noexcept {
	return std::strrchr(filepath, '.');
}
std::string getFilepathExtension(const std::string &filepath) noexcept {
	return std::string(std::strrchr(filepath.c_str(), '.'));
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


// An inclusive range of 32-bit integers.
struct CoordInclusiveRange {
	Coordinate lowerBound, upperBound;
	static constexpr Coordinate NO_MINIMUM = {INT32_MIN, INT32_MIN};
	static constexpr Coordinate NO_MAXIMUM = {INT32_MAX, INT32_MAX};

	__host__ __device__ constexpr CoordInclusiveRange() noexcept : lowerBound(CoordInclusiveRange::NO_MINIMUM), upperBound(CoordInclusiveRange::NO_MAXIMUM) {}
	__device__ constexpr CoordInclusiveRange(const CoordInclusiveRange &other) noexcept : lowerBound(other.lowerBound), upperBound(other.upperBound) {}
	__device__ constexpr CoordInclusiveRange(Coordinate value) noexcept : lowerBound(value), upperBound(value) {}
	__device__ constexpr CoordInclusiveRange(Coordinate lowerBound, Coordinate upperBound) noexcept : lowerBound{constexprMin(lowerBound.x, upperBound.x), constexprMin(lowerBound.z, upperBound.z)}, upperBound{constexprMax(lowerBound.x, upperBound.x), constexprMax(lowerBound.z, upperBound.z)} {}
	// Initialize based on the intersection of two ranges.
	__device__ constexpr CoordInclusiveRange(const CoordInclusiveRange &range1, const CoordInclusiveRange &range2) noexcept : lowerBound{constexprMax(range1.lowerBound.x, range2.lowerBound.x), constexprMax(range1.lowerBound.z, range2.lowerBound.z)}, upperBound{constexprMin(range1.upperBound.x, range2.upperBound.x), constexprMin(range1.upperBound.z, range2.upperBound.z)} {}

	// Returns if a value falls within the range.
	__host__ __device__ constexpr bool contains(Coordinate value) const noexcept {
		return this->lowerBound.x <= value.x && value.x <= this->upperBound.x && this->lowerBound.z <= value.z && value.z <= this->upperBound.z;
	}

	// Returns the range's range.
	__host__ __device__ constexpr Coordinate getRange() const noexcept {
		return {this->upperBound.x - this->lowerBound.x + 1, this->upperBound.z - this->lowerBound.z + 1};
	}
};
#endif