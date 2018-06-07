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

using record_count_t         = uint32_t;

namespace compressed {

using ship_date_t            = uint16_t;
using discount_t             = uint8_t;
using extended_price_t       = uint32_t;
using tax_t                  = uint8_t;
using quantity_t             = uint8_t;
using return_flag_t          = uint8_t;
using line_status_t          = uint8_t;


} // namespace compressed

using bit_container_t             = cuda::native_word_t;
static_assert(std::is_same<bit_container_t,uint32_t>{}, "Expecting the bit container to hold 32 bits");

