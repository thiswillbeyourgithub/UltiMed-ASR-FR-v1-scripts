<drugs_addendum>
This addendum is appended after the base rules above when the input Term is a
drug (a substance name, INN, or brand name). Everything in the base prompt still
applies without exception: the five `<must>` rules, the banned characters, the
spelled-out units, the digits/Roman-numeral conventions, and the exact
`<t>...</t>` output format. This block only re-steers the ROLE and the choice of
clinical CONTEXTS so the variants read like real medication dictation.

<role_override>
The Term is a medication. Each variant is a short French clinical text a
clinician would type or dictate WHILE PRESCRIBING, ADJUSTING, RENEWING, STOPPING
or MONITORING that drug. Write it exactly as it would appear in a prescription
line, a consultation note, or a discharge letter, never as a pharmacology lesson
or a definition of the molecule.
</role_override>

<drug_specifics>
- Name casing follows the base `<capitalization>` rule: brand names capitalized
  (Doliprane, Lévothyrox), INN and generic lowercase (paracétamol, amoxicilline).
  Keep the Term as given, only fixing its case to match this rule.
- Vary the pharmaceutical PRESENTATION across variants: form (comprimé, gélule,
  solution buvable, patch, ampoule injectable, sirop, suppositoire), dose, route,
  and frequency. Realistic doses for that drug only.
- Spell every dose and unit out per the base `<units>` rule: "cinq cents
  milligrammes" or "500 milligrammes" as digits plus the spelled-out unit, never
  "500 mg". Frequencies spoken naturally: "matin et soir", "trois fois par jour",
  "une fois par semaine".
- If a Definition line gives available forms or dosages, treat it as a factual
  hint about realistic presentations, not as text to paraphrase.
- One variant may mention only the substance name with no specific form or dose
  (e.g. a tolerance note or a treatment-history line).
</drug_specifics>

<diversification_override>
Replace the base `<diversification>` context list with this drug-oriented one.
When N > 1, cycle these in order, restarting at 1 after context 8, and never
repeat the previous variant's context:

1. new prescription
2. dose increase or decrease
3. side-effect report and management
4. renewal of an existing prescription
5. discontinuation or switch to another drug
6. biological or clinical monitoring under treatment
7. medication reconciliation on admission or discharge
8. patient history or tolerance note

Keep varying length as the base prompt requires: mix short single-line
prescription-style variants with longer 3 to 5 sentence notes, and vary the
surrounding numbers, routes and frequencies.
</diversification_override>

<drug_worked_examples>
CORRECT: N=2 for a generic INN (contexts: new prescription, dose increase).
Input:
  Term: ramipril
  Produce 2 variants.
Output:
<t>Ajout de ramipril 5 milligrammes le matin pour une hypertension artérielle mal contrôlée, contrôle de la tension et du ionogramme dans 15 jours.</t>
<t>Bonne tolérance du ramipril à 5 milligrammes, la tension reste supérieure à 140 millimètres de mercure. On passe à 10 milligrammes le matin, surveillance de la kaliémie et de la créatinine.</t>

CORRECT: N=1 for a brand name, capitalized per the base rule.
Input:
  Term: Lévothyrox
  Definition: Comprimés dosés à 25, 50, 75 et 100 microgrammes.
  Produce 1 variants.
Output:
<t>Arrêt du Lévothyrox 25 microgrammes et passage à 50 microgrammes une prise le matin à jeun, nouveau dosage de la TSH dans 6 semaines.</t>
</drug_worked_examples>
</drugs_addendum>
