#!/bin/bash

grep "#:" ex*/* | sed -E "s/(.*#: ex.*)/\n\1 /g"
echo

