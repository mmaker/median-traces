#!/bin/sh

DPATH="$1"
LEARN="python median_traces.py learn --path=$DPATH"
TEST="python median_traces.py test --path=$DPATH"

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


$TEST mf3 ori --cls mf3 "$DPATH/mf3-anti"
$TEST mf3 gau --cls mf3 "$DPATH/mf3-anti"
$TEST mf3 res --cls mf3 "$DPATH/mf3-anti"
$TEST mf3 ave --cls mf3 "$DPATH/mf3-anti"

$TEST mf5 ori --cls mf5 "$DPATH/mf5-anti"
$TEST mf5 gau --cls mf5 "$DPATH/mf5-anti"
$TEST mf5 res --cls mf5 "$DPATH/mf5-anti"
$TEST mf5 ave --cls mf5 "$DPATH/mf5-anti"

$TEST mf35     all     --cls mf35     "$DPATH/mf35-anti"
$TEST mf35-64  all-64  --cls mf35-64  "$DPATH/mf35-64-anti"
$TEST mf35-256 all-256 --cls mf35-256 "$DPATH/mf35-256-anti"
