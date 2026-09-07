# Acronyms addendum (sigles stage)

Everything in the base prompt above still applies: the exact `<t>...</t>` output format, the forbidden characters, the spelled-out units, the length rules and the diversification rules. This addendum only re-steers the role for medical acronyms and defines the extra input line this stage sends.

<role_override>
The Term is a medical acronym or sigle in common French clinical use (ESAT, HTAP, aVf, gamma-GT, ADAMTS-13). You still write short, natural French dictation sentences a clinician would say aloud in consultations, ward rounds, on-call handovers or report dictation, and the Term must appear in every variant written EXACTLY as given: same capitalization, same hyphens, same digits (aVf stays aVf, never AVF; gamma-GT stays gamma-GT). Never replace the acronym by its full expansion, never re-case it, never add or remove dots or hyphens, and never pluralize it with an "s": French sigles are invariant ("les ECG"). Any capitalization rule from the base prompt yields to the acronym's own written form.
</role_override>

<must>The input contains a line `Pronounced as: "..."` showing how the text-to-speech engine will SPEAK the acronym: dashes separate spoken chunks, so "E-S-A-T" is spelled letter by letter, "e-sath" is read as one word, and "gamma-G-T" says the word "gamma" then the letters G and T. Use it ONLY to choose articles, elision, prepositions and agreement so the sentence stays natural when the acronym is spoken that way: write l'ESAT when the spoken form opens on a vowel sound, le SAMU when it opens on a consonant. NEVER write the pronunciation itself in a variant; the written text always carries the Term exactly as given.</must>

<must>Exactly one variant must also state the meaning of the acronym in natural French, woven in the way a clinician would gloss it: an apposition, "c'est-à-dire", or a relative clause. If the Definition is in English, render that gloss in natural French rather than copying the English words. The other variants must NOT contain the expansion; they rely on clinical context alone.</must>

## Worked example (letter-spelled acronym)

Input:

    Term: ESAT
    Definition: établissement et service d'accompagnement par le travail
    Pronounced as: "E-S-A-T"
    Produce 3 variants.

Good output:

    <t>Le patient a repris une activité encadrée à l'ESAT depuis le mois de mars.</t>
    <t>Une orientation vers l'ESAT, c'est-à-dire un établissement et service d'accompagnement par le travail, a été validée en commission.</t>
    <t>À l'entretien, il décrit un bon investissement de son poste en ESAT malgré la persistance du traitement neuroleptique.</t>

Why this is good: the acronym appears verbatim in each variant; "l'ESAT" elides because the spoken form "E-S-A-T" opens on a vowel sound; exactly one variant carries the French meaning; the three clinical contexts differ.

## Worked example (acronym read as a word)

Input:

    Term: SAMU
    Definition: service d'aide médicale urgente
    Pronounced as: "samu"
    Produce 3 variants.

Good output:

    <t>La patiente a été adressée par le SAMU pour une douleur thoracique constrictive apparue au repos.</t>
    <t>Le SAMU, le service d'aide médicale urgente, a été déclenché devant les troubles de la conscience.</t>
    <t>Devant la désaturation brutale, le médecin régulateur du SAMU décide d'un transport médicalisé vers la réanimation.</t>

Why this is good: "le SAMU" takes the plain article because the spoken form opens on a consonant; exactly one variant glosses the meaning; the contexts differ.
