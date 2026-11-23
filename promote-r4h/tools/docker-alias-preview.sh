#!/bin/bash
# Preview only (GLM53_PROMOTE_TEST=1 DOCKER=<this>): the alias tag RUNBOOK.md step 2 creates does not exist yet; this maps it to the
# candidate tag it will name and passes every docker call through unchanged otherwise. Never used by a real promotion.
args=(); for arg in "$@"; do [ "$arg" = "$ALIAS_TAG" ] && arg="$REAL_TAG"; args+=("$arg"); done
exec docker "${args[@]}"
