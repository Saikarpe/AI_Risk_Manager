# 5-Minute Pitch Video — Script & Shot List

Total: 5:00. Every number below is pulled directly from `model/metrics.json` — say
these exact figures on camera, don't round further or embellish.

---

## 1. Frame the problem — 0:00–0:30 (talking head or slide)

> "When a customer returns an order, the merchant has already paid for shipping,
> paid for restocking, and paid payment-processing fees — before they ever find out
> the order was going to come back. Razorpay merchants have no way to know, at the
> moment of checkout, which orders are likely to become returns. This tool scores
> every order for return risk *before* fulfillment, so a merchant can choose to
> intervene — manual verification, delayed capture, address confirmation — on only
> the highest-risk orders."

**On screen:** title card, no logos/buzzwords.

## 2. Live demo — 0:30–2:00 (screen recording, Lovable app)

Open https://learning-horizon-helper.lovable.app live on screen.

- Show the order feed table loading and scoring a full batch (not one example) —
  point out the mix of green/red flags across ~40 orders.
- Sort by risk score descending — point out the top few are genuinely the riskiest
  profile: new customer, COD, high-risk pincode, late-night order.
- Move the threshold slider from 0.53 down to ~0.3 and back up to ~0.7 — narrate what
  happens: "as I lower the threshold, more orders get flagged — recall goes up, but
  so does the false-flag rate. At 0.53, the number we shipped, we're flagging about
  21% of order volume."
- Point at the metrics panel on screen (should already show PR-AUC/ROC-AUC/precision/recall).

> "This isn't one cherry-picked order — it's live inference on a full held-out batch,
> talking to a real deployed model over a real API."

## 3. Show the metrics honestly — 2:00–3:00 (screen: metrics panel or metrics.json)

State plainly, no hedging:

> "On a held-out test set of 1,600 orders the model never saw during training —
> PR-AUC is 0.249, against a 9.4% base return rate, so about a 2.6x lift over random.
> ROC-AUC is 0.726. At our chosen threshold of 0.53: precision is 23.5%, recall is
> 53.6%. That means 3 out of 4 flagged orders turn out fine — and we still miss
> about 46% of actual returns. That's the honest number. We didn't pick a threshold
> that maximizes accuracy — accuracy is meaningless on a 9% base rate. We built a
> cost model instead: a missed return costs the merchant a fixed handling fee plus
> 12% of order value in eaten shipping, restocking, and payment fees. A false flag
> costs a fixed verification fee plus an 8% chance the added friction makes a
> genuinely good customer abandon the order — a lost sale. Both costs scale with
> order value, on purpose. Pure cost-minimization with no constraint actually
> degenerates — it swings to flagging almost everyone or almost nobody depending on
> the exact cost ratio, and neither is useful to a merchant. So we added a floor:
> catch at least half of all risky orders, and minimize cost subject to that. That's
> how we landed on 0.53, and it saves an estimated ₹22,800 in expected cost across
> the test set versus just defaulting to 0.5."

**On screen:** show `model/metrics.json` or the metrics panel with these exact numbers
visible, so it's verifiably not made up on the spot.

## 4. Walk the architecture — 3:00–4:00 (architecture diagram)

Show `docs/architecture.svg` full-screen.

> "Four pieces. A synthetic order generator bakes in real-world correlations — new
> customer plus high order value plus a risky delivery pincode compounds risk, cash
> on delivery correlates with return-to-origin, category and time-of-purchase matter
> too — calibrated to a realistic 9.4% base return rate, not 50/50. That trains an
> XGBoost classifier with class weighting for the imbalance. The model is served by
> a FastAPI backend deployed on Render, with a `/score` endpoint and a `/metrics`
> endpoint. The Lovable frontend you just saw calls that API directly over `fetch` —
> it has no backend of its own. Everywhere in this pipeline — the CSV columns, the
> API's JSON fields, the UI's table headers — uses the exact same field names, locked
> in one schema file from day one."

## 5. One honest failure case — 4:00–4:30 (docs/failure_cases.md, or the flagged order in the UI)

Pick ONE of the two documented cases (the false positive reads better on camera —
it's a clean, relatable story):

> "Here's a case the model gets wrong. Order 101436 — a returning customer with 5
> prior orders, paying by UPI, not COD — trust signals across the board. But it's a
> ₹20,614 electronics order shipping to a pincode we've tagged high-risk, ordered at
> midnight. The model scores it 0.857 and flags it. The order was actually fine. The
> model over-weighted the pincode and the late-night signal and didn't let the
> customer's own track record override it enough. In production that's a real cost —
> a loyal customer sent through extra verification because of where they live, not
> who they are. It's also the argument for why this system only flags and
> recommends — it never auto-cancels. A human, or a downstream rule like 'never
> hard-block a customer with 5+ prior orders,' makes the final call."

## 6. Close — 4:30–5:00

> "Full code — data generation, training, the API, the deployment configs — is on
> GitHub at github.com/Saikarpe/AI_Risk_Manager. This turns return risk from
> something a merchant discovers after the fact into something they can act on
> before it costs them anything."

**On screen:** repo URL, large and readable, held for at least 3 seconds.

---

## Recording checklist

- [ ] Hit https://ai-risk-manager-kl0x.onrender.com/health ~1 minute before recording
      to wake the free-tier backend (cold start is 30-60s otherwise — don't let that
      happen live on camera).
- [ ] Refresh the Lovable app once after that so it re-scores against a warm backend.
- [ ] Screen-record at 1080p minimum; the metrics panel numbers need to be legible.
- [ ] Say the numbers exactly as in `model/metrics.json` — don't round differently
      between the video and the README, judges may cross-check.
