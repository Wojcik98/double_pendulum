#!/usr/bin/env bash
set -e

. ~/.just-completions.bash

set +e
exec "$@"
