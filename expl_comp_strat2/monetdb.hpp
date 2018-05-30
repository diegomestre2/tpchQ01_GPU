#ifndef H_MONETDB
#define H_MONETDB

/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

extern "C" {

int
todate(int day, int month, int year);

void
fromdate(int n, int* d, int* m, int* y);

}


#endif