#!/bin/bash
# Applies 027_unmix_installs.sql to the PRODUCTION database, using the DATABASE_URL secret inside the Fly machine
# (a local backend/.env points at a local dev database, so it cannot be used for this).
# The migration is idempotent (IF NOT EXISTS), so running it twice is harmless.
set -euo pipefail
fly ssh console -a bandos-backend -C "node -e \"require('./src/db/client').query(require('fs').readFileSync('src/db/migrations/027_unmix_installs.sql','utf8')).then(()=>{console.log('applied');process.exit(0)}).catch(e=>{console.error(e.message);process.exit(1)})\""
