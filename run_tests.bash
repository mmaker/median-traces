#!/bin/sh

LEARN="python median_traces.py learn"

$LEARN mf5 ori
$LEARN mf5 gau
$LEARN mf5 res
$LEARN mf5 ave

$LEARN mf3 ori
$LEARN mf3 gau
$LEARN mf3 res
$LEARN mf3 ave

$LEARN mf35     all
$LEARN mf35-64  all-64
$LEARN mf35-256 all-256
