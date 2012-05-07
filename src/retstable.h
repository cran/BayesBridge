// This code has been taken from the copula package and slightly modified for
// our use.  It is licensed under GPL 3.

#ifndef __RRETSTABLE__
#define __RRETSTABLE__

#include "RNG.h"

// Helpers.
double sinc_MM(double x);
double A_(double x, double alpha);
double BdB0(double x,double alpha);

// Stable draw by Devroye.
double retstable_LD(double h, double alpha, RNG& r, double V0=1.0);

#endif
