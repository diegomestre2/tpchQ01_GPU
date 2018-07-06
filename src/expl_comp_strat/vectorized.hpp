#ifndef H_vectorized
#define H_vectorized

// #define DEBUG

#include "common.hpp"

#include <cassert>
#include <stdint.h>
#include <string>
#include <cstring>
#include <x86intrin.h>

struct Primitives {
    template<typename F>
    static int for_each(const sel_t* RESTRICT sel, int n, F&& fun) {
        int i;
        if (sel) {
#define a(dx) fun(sel[i + dx]);
            for (i=0; i+8<n; i+=8) {
                a(0); a(1); a(2); a(3);
                a(4); a(5); a(6); a(7);
            }
#undef a

            for (; i<n; i++) {
                fun(sel[i]);
            }
        } else {
            for (i=0; i<n; i++) {
                fun(i);
            }
        }
        return n;
    }

    template<typename F>
    static int for_each_scalar(const sel_t* RESTRICT sel, int n, F&& fun) {
        int i;
        if (sel) {
            for (i=0; i<n; i++) {
                fun(sel[i]);
            }
        } else {
            for (i=0; i<n; i++) {
                fun(i);
            }
        }
        return n;
    }

    template<typename T, typename F>
    static auto map(T* RESTRICT out, const sel_t* RESTRICT sel, int n, F&& fun) {
        return for_each(sel, n, [&] (auto i) { out[i] = fun(i); });
    }

    template<typename F>
    static int select(sel_t* RESTRICT out, const sel_t* RESTRICT sel, int n, bool data_dep, F&& fun) {
        int num = 0;
        if (data_dep) {
            for_each(sel, n, [&] (int idx) { int r = fun(idx); out[num] = idx; num += r; });
        } else {
            for_each(sel, n, [&] (int idx) { if (fun(idx)) { out[num++] = idx; } });
        }
        
        return num;
    }

    template<typename F, typename G>
    static void for_each_2(const sel_t* RESTRICT sel, int n, F&& fun, G&& remainder) {
        const int m = n & ~1;
        if (sel) {
            int k;
            for (k=0; k<m; k+=2) {
                fun(sel[k], sel[k+1], k, k+1);
            }

            if (k < n) {
                remainder(sel[k], k);
            }
        } else {
            int i;
            for (i=0; i<m; i+=2) {
                fun(i, i+1, i, i+1);
            }

            if (i < n) {
                remainder(i, i);
            }
        }
    }

    template<typename T>
    static T*
    desel(T* RESTRICT out, const sel_t* RESTRICT sel, T* RESTRICT col, int num)
    {
        if (!sel) {
            return col;
        }

        for(int i=0; i<num; i++) {
            out[i] = col[sel[i]];
        }
        return out;
    }


    template<typename InitAggrFun, typename UpdateAggrFun, typename FinalizeAggrFun>
    static int ordaggr(const idx_t* RESTRICT pos, const idx_t* RESTRICT lim, const idx_t* RESTRICT grp, idx_t num_groups,
            InitAggrFun&& init_aggr, UpdateAggrFun&& upd_aggr, FinalizeAggrFun&& fin_aggr) {
        size_t k=0, i=0;
        if (pos) {
            for (idx_t g=0; g<num_groups; g++) {
                init_aggr(g);
                while (k < lim[g]) {
                    upd_aggr(g, pos[k]);
                    k++;
                }
                fin_aggr(g);
            }
        } else {
            for (idx_t g=0; g<num_groups; g++) {
                init_aggr(g);
                while (i < lim[g]) {
                    upd_aggr(g, i);
                    i++;
                }
                fin_aggr(g);
            }
        }

        return 0;
    }

    // remembers first k of group
    template<typename InitAggrFun, typename UpdateAggrFun, typename FinalizeAggrFun>
    static int ordaggr_fst(const idx_t* RESTRICT pos, const idx_t* RESTRICT lim, const idx_t* RESTRICT grp, idx_t num_groups,
            InitAggrFun&& init_aggr, UpdateAggrFun&& upd_aggr, FinalizeAggrFun&& fin_aggr) {
        size_t k=0, i=0;
        if (pos) {
            for (idx_t g=0; g<num_groups; g++) {
                idx_t first = k;
                init_aggr(g);
                while (k < lim[g]) {
                    upd_aggr(g, pos[k]);
                    k++;
                }
                fin_aggr(g, first);
            }
        } else {
            for (idx_t g=0; g<num_groups; g++) {
                idx_t first = k;
                init_aggr(g);
                while (i < lim[g]) {
                    upd_aggr(g, i);
                    i++;
                }
                fin_aggr(g, first);
            }
        }

        return 0;
    }

public:
    static int NOINL partial_shuffle_scalar(idx_t* RESTRICT gids, sel_t* RESTRICT aggr_sel, int num, idx_t* RESTRICT sel,
        idx_t* RESTRICT lim, idx_t* RESTRICT grp, uint16_t** RESTRICT grppos0, uint16_t** RESTRICT grppos1,
        uint16_t* RESTRICT selbuf0, uint16_t* RESTRICT selbuf1);

    static int NOINL partial_shuffle_avx512_cmp(idx_t* RESTRICT gids, sel_t* RESTRICT aggr_sel, int num, idx_t* RESTRICT sel,
        idx_t* RESTRICT lim, idx_t* RESTRICT grp, uint16_t** RESTRICT grppos0, uint16_t** RESTRICT grppos1,
        uint16_t* RESTRICT selbuf0, uint16_t* RESTRICT selbuf1);

    /** Improved version using a in-register lookup table and population counts */
    static int NOINL partial_shuffle_avx512(idx_t* RESTRICT gids, sel_t* RESTRICT aggr_sel, int num, idx_t* RESTRICT sel,
        idx_t* RESTRICT lim, idx_t* RESTRICT grp, uint16_t** RESTRICT grppos0, uint16_t** RESTRICT grppos1, 
        uint16_t* RESTRICT selbuf0, uint16_t* RESTRICT selbuf1);


    static uint64_t NOINL map_charge(int64_t* RESTRICT res, int32_t*RESTRICT col1, int8_t*RESTRICT col2, sel_t*RESTRICT sel, int n);

    static int NOINL select_int32_t(sel_t* RESTRICT out, sel_t* RESTRICT sel, int n, bool data_dep, int* RESTRICT a, int b);
    static int NOINL select_int16_t(sel_t* RESTRICT out, sel_t* RESTRICT sel, int n, bool data_dep, int16_t* RESTRICT a, int16_t b);
    static int NOINL select_int16_t_avx512(sel_t* RESTRICT out, sel_t* RESTRICT sel, int n, bool data_dep, int16_t* RESTRICT a, int16_t b);


    static int NOINL map_gid2_dom_restrict(idx_t* RESTRICT out, sel_t* RESTRICT sel, int n, int8_t* RESTRICT a, int8_t min_a, int8_t max_a, int8_t* RESTRICT b, int8_t min_b, int8_t max_b);
    static int NOINL map_gid(idx_t* RESTRICT out, sel_t* RESTRICT sel, int n, int8_t* RESTRICT a, int8_t* RESTRICT b);

    static int NOINL map_disc_1(int8_t* RESTRICT out, sel_t* RESTRICT sel, int n, int8_t RESTRICT b, int8_t* RESTRICT a);

    static int NOINL map_tax_1(int8_t* RESTRICT out, sel_t* RESTRICT sel, int n, int8_t* RESTRICT a, int8_t RESTRICT b);


    static int NOINL map_disc_price(int32_t* RESTRICT out, sel_t* RESTRICT sel, int n, int8_t* RESTRICT a, int32_t* RESTRICT b);

    static int NOINL ordaggr_quantity(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int16_t* RESTRICT quantity);
    static int NOINL ordaggr_extended_price(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int32_t* RESTRICT quantity);
    static int NOINL ordaggr_disc_price(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int32_t* RESTRICT quantity);
    static int NOINL ordaggr_charge(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int64_t* RESTRICT charge);
    static int NOINL ordaggr_disc(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int8_t* RESTRICT charge);
    static int NOINL ordaggr_count(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups);

    static int NOINL par_ordaggr_quantity(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int16_t* RESTRICT quantity);
    static int NOINL par_ordaggr_extended_price(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int32_t* RESTRICT quantity);
    static int NOINL par_ordaggr_disc_price(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int32_t* RESTRICT quantity);
    static int NOINL par_ordaggr_charge(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int64_t* RESTRICT charge);
    static int NOINL par_ordaggr_disc(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int8_t* RESTRICT charge);

    static int NOINL ordaggr_all_in_one(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int16_t* RESTRICT quantity, int32_t* RESTRICT price, int32_t* RESTRICT disc_price, int64_t* RESTRICT charge, int8_t* RESTRICT disc);



    static int NOINL old_map_gid(idx_t* RESTRICT out, sel_t* RESTRICT sel, int n, char* RESTRICT v_returnflag, char* RESTRICT v_linestatus);
    static int NOINL old_map_disc_1(int64_t* RESTRICT v_disc_1, sel_t* RESTRICT sel, int n, int64_t dec_one, int64_t* RESTRICT v_discount);

    static int NOINL old_map_tax_1(int64_t* RESTRICT v_disc_1, sel_t* RESTRICT sel, int n, int64_t dec_one, int64_t* RESTRICT v_tax);
    static int NOINL old_map_mul(int64_t* RESTRICT v_disc_price, sel_t* RESTRICT sel, int n, int64_t* RESTRICT v_disc_1, int64_t* RESTRICT v_extendedprice);

    static int NOINL old_ordaggr_all_in_one(AggrHashTable* RESTRICT aggr0, idx_t* RESTRICT pos, idx_t* RESTRICT lim, idx_t* RESTRICT grp, idx_t num_groups, int64_t* RESTRICT quantity, int64_t* RESTRICT extendedprice, int64_t* RESTRICT disc_price, int64_t* RESTRICT charge, int64_t* RESTRICT disc_1);
};

#endif