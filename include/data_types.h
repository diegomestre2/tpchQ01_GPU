#pragma once

#include <cstdint>
#include <cstddef>

using std::size_t;

using ship_date_t            = int32_t;
using discount_t             = int64_t;
using extended_price_t       = int64_t;
using tax_t                  = int64_t;
using quantity_t             = int64_t;
using return_flag_t          = int8_t;
using line_status_t          = int8_t;

using sum_quantity_t         = uint64_t;
using sum_base_price_t       = uint64_t;
using sum_discounted_price_t = uint64_t;
using sum_charge_t           = uint64_t;
using sum_discount_t         = uint64_t;

using cardinality_t         = uint32_t; // Limiting ourselves to SF 500 here

namespace compressed {

using ship_date_t            = uint16_t;
using discount_t             = uint8_t;
using extended_price_t       = uint32_t;
using tax_t                  = uint8_t;
using quantity_t             = uint8_t;
using return_flag_t          = uint8_t; // Don't use this!
using line_status_t          = uint8_t; // Don't use this!
} // namespace compressed

using bit_container_t             = cuda::native_word_t;
static_assert(std::is_same<bit_container_t,uint32_t>{}, "Expecting the bit container to hold 32 bits");

/**
 * Applies a DICT(1 bit) encoding scheme to line status values
 */
inline uint8_t encode_line_status(char status)
{
#ifdef NDEBUG
	return (status == 'F') ? 1 : 0;
#else
	switch(status) {
	case 'F': return 0;
	case 'O': return 1;
	default: throw std::invalid_argument("No such line status '" + status + "'");
	}
#endif
}

/**
 * Applies a DICT(2 bit) encoding scheme to return flag values
 */
inline uint8_t encode_return_flag(char flag)
{
	switch(flag) {
	case 'N' : return 0b00;
	case 'R' : return 0b01;
#ifdef NDEBUG
	default:   return 0b10;
#else
	case 'A' : return 0b10;
	default: throw std::invalid_argument("No such return flag '" + flag + "'");
#endif
	}
}

inline char decode_line_status(uint8_t encoded_status)
{
#ifdef NDEBUG
	return encoded_status == 0 ? 'F' : 'O';
#else
	switch(encoded_status) {
	case 0b0: return 'F';
	case 0b1: return 'O';
	default: throw std::invalid_argument("No such encoded line status '" + encoded_status + "'");
	}
#endif
}

/**
 * Applies a DICT(2 bit) encoding scheme to return flag values
 */
inline uint8_t decode_return_flag(char encoded_flag)
{
	switch(encoded_flag) {
	case 0b00: return 'N';
	case 0b01: return 'R';
#ifdef NDEBUG
	default:   return 'A';
#else
	case 0b10: return 'A';
	default: throw std::invalid_argument("No such encoded return flag '" + encoded_flag + "'");
#endif
	}
}


