#!/bin/bash
(cd nlopt; sh BUILD.gcc.sh)&
(cd adolc; sh BUILD.gcc.sh)&
wait

