/*
 Portions Copyright (c) 1996-2010, PostgreSQL Global Development Group
 Portions Copyright (c) 1994, Regents of the University of California

 This file is licensed under a BSD license.
*/

#ifndef DT_GETRUSAGE_H
#define DT_GETRUSAGE_H

#include <sys/time.h> /* for struct timeval */

struct rusage {
    struct timeval ru_utime; /* user CPU time used */
    struct timeval ru_stime; /* system CPU time used */
    long   ru_maxrss;        /* maximum resident set size */
    long   ru_ixrss;         /* integral shared memory size */
    long   ru_idrss;         /* integral unshared data size */
    long   ru_isrss;         /* integral unshared stack size */
    long   ru_minflt;        /* page reclaims (soft page faults) */
    long   ru_majflt;        /* page faults (hard page faults) */
    long   ru_nswap;         /* swaps */
    long   ru_inblock;       /* block input operations */
    long   ru_oublock;       /* block output operations */
    long   ru_msgsnd;        /* IPC messages sent */
    long   ru_msgrcv;        /* IPC messages received */
    long   ru_nsignals;      /* signals received */
    long   ru_nvcsw;         /* voluntary context switches */
    long   ru_nivcsw;        /* involuntary context switches */
};

#ifndef RUSAGE_SELF
#define RUSAGE_SELF 1
#endif

#ifndef RUSAGE_CHILDREN
#define RUSAGE_CHILDREN 2
#endif

#ifndef RUSAGE_THREAD
#define RUSAGE_THREAD 3
#endif

extern int getrusage(int who, struct rusage *rusage);

#endif // DT_GETRUSAGE_H

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
