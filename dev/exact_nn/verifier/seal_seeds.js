// seal_seeds.js — independent-verifier sealed seed generation (SPEC §4c).
//
// The program splits the seed line three ways so τ calibration cannot peek at
// the verification set (anti-circularity):
//
//   [0, 270000)        TRAIN       — side-a student training shards
//   [270000, 280000)   CALIBRATION — published; τ is frozen against THIS set
//                                     (prod-vs-student disagreement is "spent" here)
//   [290000, 2^31)     SEALED pool — side-b draws the sealed verification seeds
//                                     from here with a SECRET salt; never revealed
//                                     until the one-shot verdict is posted.
//   [280000, 290000)   buffer/dev  — harness self-test, smoke (no tuning, no verdict)
//
// Sealing mechanism: sealed[i] = FLOOR + ( HMAC_SHA256(salt, "sealed:"+i)
// mod (2^31 − FLOOR) ), deduped, for i = 0.. until COUNT distinct seeds. The
// salt (32 random bytes) lives ONLY at ~/.exactnn_seal_salt (chmod 600, never
// committed, never in /shared — /shared is visible to side-a). What IS
// committed is the COMMITMENT: sha256(salt) + {FLOOR, COUNT, algorithm}. That
// pre-registers the sealed set (side-a cannot target seeds it cannot derive)
// AND lets anyone later verify side-b didn't move the goalposts: at verdict
// time `--reveal` prints the seeds and the salt whose sha256 must equal the
// committed value.
//
//   node seal_seeds.js --init                 # one-time: make salt + commitment
//   node seal_seeds.js --list [--count N]      # print sealed seeds (verifier only)
//   node seal_seeds.js --reveal                # print salt + seeds (verdict/audit)
//   node seal_seeds.js --verify <commit.json>  # check salt matches a commitment
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

// env-overridable so a fresh sealed pool can be minted per phase WITHOUT clobbering
// a prior phase's committed artifacts (Phase-2 re-seal: the Phase-1 salt was revealed
// in evidence/seal_reveal_audit_trail.json → no longer secret; a credible one-shot
// needs a fresh, never-revealed salt).
const SALT_PATH = process.env.EXACTNN_SALT_PATH || path.join(os.homedir(), '.exactnn_seal_salt');
const COMMIT_PATH = process.env.EXACTNN_COMMIT_PATH || path.join(__dirname, 'seal_commitment.json');
const FLOOR = 290000;
const POOL_HI = 2 ** 31;               // exclusive
const COUNT = 4096;                    // sealed seeds (sized for rule-of-three margin)
const ALGO = 'HMAC-SHA256(salt, "sealed:"+i) mod (2^31-FLOOR) + FLOOR, dedup, i=0..';

function loadSalt() {
    if (!fs.existsSync(SALT_PATH)) throw new Error('no salt at ' + SALT_PATH + ' — run --init');
    return fs.readFileSync(SALT_PATH);
}

function sealedSeeds(salt, count) {
    const span = POOL_HI - FLOOR;
    const seen = new Set(), out = [];
    for (let i = 0; out.length < count; i++) {
        const h = crypto.createHmac('sha256', salt).update('sealed:' + i).digest();
        // take the low 6 bytes as an unsigned int, mod span
        const v = Number(h.readBigUInt64BE(0) & 0xffffffffffffn) % span + FLOOR;
        if (!seen.has(v)) { seen.add(v); out.push(v); }
        if (i > count * 100) throw new Error('dedup runaway');
    }
    return out;
}

function commitmentFor(salt) {
    return {
        sha256_salt: crypto.createHash('sha256').update(salt).digest('hex'),
        FLOOR, COUNT, POOL_HI, algorithm: ALGO,
        note: 'Sealed verification seeds for EXACT-NN (SPEC §4c). Salt is held only '
            + 'by side-b (~/.exactnn_seal_salt, chmod 600). This commitment pre-registers '
            + 'the sealed set without revealing it. At verdict, `seal_seeds.js --reveal` '
            + 'prints the salt; sha256(salt) must equal sha256_salt above.',
    };
}

function cmdInit() {
    if (fs.existsSync(SALT_PATH)) {
        console.error('salt already exists at ' + SALT_PATH + ' — refusing to overwrite '
            + '(delete it manually to re-seal; this changes the commitment).');
        process.exit(1);
    }
    const salt = crypto.randomBytes(32);
    fs.writeFileSync(SALT_PATH, salt, { mode: 0o600 });
    fs.chmodSync(SALT_PATH, 0o600);
    const commit = commitmentFor(salt);
    fs.writeFileSync(COMMIT_PATH, JSON.stringify(commit, null, 2) + '\n');
    console.log('sealed: salt written ' + SALT_PATH + ' (chmod 600)');
    console.log('committed ' + COMMIT_PATH + ':');
    console.log(JSON.stringify(commit, null, 2));
}

function cmdList(count) {
    const salt = loadSalt();
    const seeds = sealedSeeds(salt, count || COUNT);
    process.stdout.write(seeds.join('\n') + '\n');
}

function cmdReveal() {
    const salt = loadSalt();
    const seeds = sealedSeeds(salt, COUNT);
    console.log(JSON.stringify({
        salt_hex: salt.toString('hex'),
        sha256_salt: crypto.createHash('sha256').update(salt).digest('hex'),
        FLOOR, COUNT, seeds,
    }, null, 1));
}

function cmdVerify(commitFile) {
    const salt = loadSalt();
    const claimed = JSON.parse(fs.readFileSync(commitFile, 'utf8'));
    const actual = crypto.createHash('sha256').update(salt).digest('hex');
    const ok = actual === claimed.sha256_salt;
    console.log((ok ? 'MATCH' : 'MISMATCH') + ': sha256(salt)=' + actual
        + ' committed=' + claimed.sha256_salt);
    process.exit(ok ? 0 : 1);
}

function main() {
    const a = process.argv.slice(2);
    if (a[0] === '--init') return cmdInit();
    if (a[0] === '--list') {
        const ci = a.indexOf('--count');
        return cmdList(ci >= 0 ? +a[ci + 1] : COUNT);
    }
    if (a[0] === '--reveal') return cmdReveal();
    if (a[0] === '--verify') return cmdVerify(a[1]);
    console.error('usage: --init | --list [--count N] | --reveal | --verify <commit.json>');
    process.exit(2);
}

module.exports = { sealedSeeds, commitmentFor, FLOOR, COUNT, SALT_PATH, COMMIT_PATH };
if (require.main === module) main();
