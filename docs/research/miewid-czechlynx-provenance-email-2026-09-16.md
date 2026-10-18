# MiewID-msv3 / CzechLynx overlap inquiry — 16 September 2026

Draft prepared by the assistant. The user subsequently reported a blocked email attempt; successful delivery and any author response remain unconfirmed.

Published contacts: Lasha Otarashvili, Lasha@conservationxlabs.org; Jason A. Holmberg, Jason@conservationxlabs.org. Rechecked against the indexed first-page author information in [On Combining Animal Re-Identification Models to Address Small Datasets](https://dspace.vut.cz/server/api/core/bitstreams/f298e51a-9ee8-4fd5-a73a-63920497e904/content). Direct PDF retrieval returned 404 during this check; the search index exposes both addresses. Deliverability has not been tested.

To: Lasha@conservationxlabs.org
Cc: Jason@conservationxlabs.org
Subject: Question about MiewID-msv3 training overlap with CzechLynx

Dear Lasha and Jason,

I’m developing an animal re-identification pipeline for wildlife monitoring in Croatia and evaluating frozen MiewID embeddings on CzechLynx. Before interpreting the results as generalisation to unseen animals, I would like to clarify whether my evaluation data overlaps with the model’s training data.

I’m using conservationxlabs/miewid-msv3 at Hugging Face revision:
4f1d7f2b521149e5fe34bb85f377248ce9971a7d

The dataset is the real-image CzechLynx release:
https://zenodo.org/records/17592004

I have not fine-tuned MiewID. My own fitting, calibration and selection pools contain different animal identities, but this cannot rule out prior exposure in the pretrained checkpoint.

Could you help clarify:

1. Did this checkpoint’s training, validation or model selection include photographs from the Friends of the Earth Czech Republic Carpathian/Bohemian collections or Šumava National Park? I’m also interested in earlier exports supplied through Wildbook/Whiskerbook or under another name, including any training inherited from earlier checkpoints.
2. If the exact photographs were not used, could the same individual lynx have appeared in other images used to develop the model?
3. Is there a source list, collection cutoff or way to check overlap that would help identify an independent CzechLynx evaluation subset?

I’m trying to distinguish unseen photographs, including alternate crops of the same originals, from genuinely unseen individuals. If the training inventory cannot be shared, a source-level clarification or private overlap check would still be very helpful. I can provide the relevant original filenames, identity labels, dates and image hashes.

Thank you for making MiewID available and for any guidance you can offer.

Best regards,
[Your name]
[Affiliation, if applicable]

## Delivery update

The user attempted email and received a policy-block rejection identifying `catch-all@conservationxlabs.org`. This establishes a failed delivery attempt, not whether both original author addresses are invalid or which routing rule caused it. The catch-all address may reflect internal routing; its configuration is not known. The assistant has not sent any messages.

The .org author addresses are printed in the [2026 publisher PDF, page 1](https://link.springer.com/content/pdf/10.1007/s11263-025-02708-9.pdf), but publication is not proof of deliverability. Do not describe them as verified working inboxes.

An alternative **published** address for Jason is `jason@conservationxlabs.com`, shown on the last slide of his [Simons Institute presentation](https://simons.berkeley.edu/sites/default/files/2024-07/Decoding%20Communication%202024%20Advances%20in%20Flukebook%20AI%20-%20Jason%20Holmberg.pdf#page=31). Delivery is untested. This source still gives Lasha's .org address, so do not invent a .com variant for Lasha.

A contact route independent of that email delivery path is the [Wild Me community forum](https://community.wildme.org/), which their [official contact page](https://www.wildme.org/contact-us.html) says staff monitor closely. The forum has an official [MiewID announcement](https://community.wildme.org/t/miewid-v4-announcement/5406). A new provenance question can use the shortened email draft and request contact with the MiewID team.

[Google's explanation](https://knowledge.workspace.google.com/admin/users/advanced/mail-policy-requirements-error) attributes this error to sender or recipient organizational mail policies; it does not establish that a mailbox does not exist.

## Internal checkpoint reference

Publisher weight-file SHA-256 from the previous audit:
adff92b39678f37eb74861c6399a741639a8907ec2382738e903d6120727b348

Current selection result is 562/881 top-1 (63.79%), with upstream overlap unverified. No new provenance conclusion is implied by the email or the locally disjoint pools.
