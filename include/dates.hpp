

/* phony zero values, used to get negative numbers from unsigned
 * sub-integers in rule */
#define WEEKDAY_ZERO	8
#define DAY_ZERO	32
#define OFFSET_ZERO	4096

typedef int date;

static const char *MONTHS[13] = {
	NULL, "january", "february", "march", "april", "may", "june",
	"july", "august", "september", "october", "november", "december"
};

static const char *DAYS[8] = {
	NULL, "monday", "tuesday", "wednesday", "thursday",
	"friday", "saturday", "sunday"
};
static const char *COUNT1[7] = {
	NULL, "first", "second", "third", "fourth", "fifth", "last"
};
static const char *COUNT2[7] = {
	NULL, "1st", "2nd", "3rd", "4th", "5th", "last"
};
__device__
static int LEAPDAYS[13] = {
	0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};
__device__
static int CUMDAYS[13] = {
	0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365
};
__device__
static int CUMLEAPDAYS[13] = {
	0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366
};

#define YEAR_MAX		5867411
#define YEAR_MIN		(-YEAR_MAX)
#define MONTHDAYS(m,y)	((m) != 2 ? LEAPDAYS[m] : leapyear(y) ? 29 : 28)
#define YEARDAYS(y)		(leapyear(y) ? 366 : 365)
#define DATE(d,m,y)		((m) > 0 && (m) <= 12 && (d) > 0 && (y) != 0 && (y) >= YEAR_MIN && (y) <= YEAR_MAX && (d) <= MONTHDAYS(m, y))
#define TIME(h,m,s,x)	((h) >= 0 && (h) < 24 && (m) >= 0 && (m) < 60 && (s) >= 0 && (s) < 60 && (x) >= 0 && (x) < 1000)
#define LOWER(c)		((c) >= 'A' && (c) <= 'Z' ? (c) + 'a' - 'A' : (c))

/*
 * auxiliary functions
 */


#define leapyear(y)		((y) % 4 == 0 && ((y) % 100 != 0 || (y) % 400 == 0))
__device__
static int
leapyears(int year)
{
	/* count the 4-fold years that passed since jan-1-0 */
	int y4 = year / 4;

	/* count the 100-fold years */
	int y100 = year / 100;

	/* count the 400-fold years */
	int y400 = year / 400;

	return y4 + y400 - y100 + (year >= 0);	/* may be negative */
}
__device__
static date
todate_(int day, int month, int year)
{
	date n = 0;

	if (DATE(day, month, year)) {
		if (year < 0)
			year++;				/* HACK: hide year 0 */
		n = (date) (day - 1);
		if (month > 2 && leapyear(year))
			n++;
		n += CUMDAYS[month - 1];
		/* current year does not count as leapyear */
		n += 365 * year + leapyears(year >= 0 ? year - 1 : year);
	}
	return n;
}