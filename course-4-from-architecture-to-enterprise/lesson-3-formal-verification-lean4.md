# Lesson 3: Formal Verification of Cryptographic Protocols with Lean 4

*Course 4: From Architecture to Enterprise*

## The Trap: "It's Formally Verified!"

The most dangerous three words in crypto security. A proof is only as good as:

1. **What property** you're proving
2. **What threat model** you're assuming
3. **What abstraction** of the protocol you're verifying

Get any wrong → beautiful, correct, *irrelevant* proof.

**Bridge analogy:** "I have a mathematical proof this bridge is safe" — safe against what? Pedestrian traffic? Freight trucks? Earthquakes? A proof it won't collapse under pedestrians is mathematically correct and completely useless if you're driving trucks across it.

## The Hierarchy of Correctness in Lean 4

### Level 1: It Compiles

Syntactically valid Lean 4. Types align. Compiler happy. LLMs do this easily.

**Proves:** Code is well-formed. Nothing more. Blueprint uses valid notation — says nothing about whether the bridge stands.

### Level 2: It Proves *Something*

Actual theorem: "sign then verify returns true." Lean accepts the proof.

**Proves:** Functions are consistent with each other. **Does NOT prove** security. A scheme returning `true` for every input is perfectly consistent — and perfectly broken.

### Level 3: It Proves a *Security* Property

Example: existential unforgeability — "no adversary can forge a signature on unseen messages."

**But under what threat model?** Can the adversary:
- Observe all traffic? (passive attacker)
- Modify messages? (active attacker)
- Compromise participants? (Byzantine model)
- Measure timing? (side-channel model)
- Use quantum computers? (post-quantum model)

Proof is only valid against the adversary model defined in the `Adversary` type. If it doesn't model side-channel attacks and the real attacker uses timing analysis, **proof is correct and protocol is broken.**

### Level 4: It Proves the Right Property Under the Right Threat Model

Requires:
- Correct **security properties** (not just one — all relevant ones)
- Correct **threat model** (matching actual deployment)
- Correct **abstraction** (Lean model faithfully represents implementation)

**This is the actual hard problem. LLMs fail spectacularly here.**

## Why LLMs Fail at Level 4

LLMs are extremely good at Levels 1-2 and dangerously bad at Level 4.

LLMs can:
- Write syntactically correct Lean (Level 1) — easily
- Construct proofs Lean accepts (Level 2) — increasingly well
- Prove standard textbook properties (Level 3) — with decent accuracy

LLMs **cannot** reliably:
- Judge whether the property is the *right* property for this protocol
- Assess whether the threat model captures *actual* deployment threats
- Verify the Lean abstraction faithfully represents the real implementation

Because these require **understanding the gap between model and reality.** LLM sees the formal model, not the deployed system, network topology, hardware, or economic incentives of attackers. Proving theorems about a *map* with no ability to check whether the map matches the *territory*.

This is the specification-level error from Course 3 Lesson 5, but hiding behind a mathematical proof — making it **more dangerous**, not less. False certainty is worse than honest uncertainty.

## How the Orthogonal Committee Solves This

Each adapter specializes in a different layer of the verification gap:

```
Adapter 1: Property Completeness
  "Are we proving ALL security properties this protocol needs?"
  "Proof covers unforgeability but NOT forward secrecy —
   and this is a messaging protocol where forward secrecy is critical."

Adapter 2: Threat Model Adequacy
  "Does the adversary model capture realistic attacks?"
  "Model assumes passive network attacker, but protocol runs on
   public blockchain where transaction ordering is adversarial (MEV)."

Adapter 3: Abstraction Fidelity
  "Does the Lean model match what the Solidity code actually does?"
  "Lean assumes atomic execution, but Solidity has a reentrancy window
   between balance check and transfer."

Adapter 4: Cryptographic Primitive Assumptions
  "Are the underlying hardness assumptions still valid?"
  "Proof assumes discrete log hardness, but the curve was recently
   shown to have a subgroup weakness."

Adapter 5: Economic / Game-Theoretic Attacks
  "Is the protocol secure against rational economic actors?"
  "Cryptographically sound but has sandwich attack vector letting
   validators extract $X per block."
```

No single reviewer covers all five. Orthogonality isn't just useful — it's **necessary**.

## The Composition Model's Unique Role

In formal verification, the composition model checks whether **the collection of proofs covers the protocol's security requirements**:

```
Adapter 1: "Unforgeability, forward secrecy, replay resistance proved."
Adapter 2: "Threat model covers network adversary + compromised participants."
Adapter 3: "Lean abstraction matches Solidity implementation."
Adapter 4: "Cryptographic assumptions current."
Adapter 5: "No obvious economic attack vectors."

Composition model: "But does anyone cover COMPOSABILITY?
This protocol interacts with three other DeFi protocols.
Security properties proved in isolation. What happens when they compose?"
```

**Composability attacks** — individually-secure protocols becoming insecure when combined — are the #1 source of DeFi exploits formal verification misses. The diagonal bug problem applied to formal methods.

---

## Q&A

**Question:** In code review, mutation testing works — inject bugs, check if committee catches them. In formal verification, how do you generate training data for catching insufficient threat models or incomplete security properties? You can't just mutate code — you have to mutate the *proof* and *specification*.

**Student's Answer:** We can still use LLMs to mutate proof and spec — the spec and proof are normally written in LaTeX, so they're text artifacts LLMs can manipulate.

**Evaluation:** Exactly right — specs and proofs are text artifacts, manipulable like code. But the *types* of mutations needed are fundamentally different.

### Three Levels of Mutation for Formal Verification

**Level 1: Mutate the Proof (Easy, Least Valuable)**
Break proof steps, insert `sorry` (Lean's "trust me"). Problem: Lean compiler flags `sorry` trivially. Interesting mutations leave proof **compiling cleanly** but proving something weaker.

**Level 2: Mutate the Specification (Harder, Very Valuable)**
LLM systematically:
- Takes complete threat model → removes one adversary capability
- Takes security property set → removes one property
- Takes an assumption → weakens it

Each produces a spec that's **internally consistent but insufficient.** Proof still checks. Question for committee: does it notice what's missing?

**Level 3: Mutate the Abstraction Gap (Hardest, Most Valuable)**
Introduce deliberate divergences between formal model and implementation. The "anti-mutation": keep the Lean model idealized while reality diverges. Proof correct, code vulnerable, gap invisible to anyone only reading Lean.

### The Killer Data Source: Historical Exploits

Better than any synthetic mutation. Crypto has hundreds of hacked protocols with detailed post-mortems:

```
1. Take protocol as it existed before hack
2. Write the formal spec that was implicitly assumed
3. Show that spec WOULD HAVE PASSED verification
4. Identify which missing property / threat model gap allowed exploit
5. Train committee: "given this spec, would you flag it as incomplete?"
```

Example: Euler Finance ($197M). Implicit spec covered basic lending invariants but didn't specify behavior when flash loan manipulates price oracle within single transaction. Threat model without atomic composability with flash loans would have verified protocol as "secure."

```
Synthetic mutations:     unlimited quantity, variable quality
Historical exploits:     limited quantity, perfect quality and relevance
Combined:                synthetic for bulk training, real incidents for calibration
```
