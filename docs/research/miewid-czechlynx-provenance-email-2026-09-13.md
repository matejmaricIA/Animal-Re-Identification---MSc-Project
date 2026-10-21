# MiewID and CzechLynx provenance email

Draft only; not sent. Public contact details checked on 13 September 2026.

| Recipient | Address | Purpose |
| --- | --- | --- |
| Lasha Otarashvili | Lasha@conservationxlabs.org | MiewID author; checkpoint training provenance |
| Jason A. Holmberg | Jason@conservationxlabs.org | MiewID coauthor; contributing Wildbook collections |
| Lukáš Picek | lukaspicek@gmail.com | CzechLynx corresponding author; source-data sharing history |

Lasha and Jason's addresses appear on page 1 of their [2026 IJCV paper](https://dspace.vut.cz/server/api/core/bitstreams/f298e51a-9ee8-4fd5-a73a-63920497e904/content). Lukáš's address is published in the [CzechLynx article](https://www.nature.com/articles/s41597-026-06853-9_reference.pdf) and its [PubMed record](https://pubmed.ncbi.nlm.nih.gov/41735338/). These are published contact addresses; deliverability has not been tested.

To: Lasha@conservationxlabs.org  
Cc: Jason@conservationxlabs.org; lukaspicek@gmail.com  
Subject: MiewID-msv3 training overlap with CzechLynx

Dear Lasha, Jason and Lukáš,

I’m developing an animal re-identification pipeline for wildlife monitoring in Croatia and evaluating it on CzechLynx before approaching potential conservation partners.

Using frozen conservationxlabs/miewid-msv3 embeddings on the supplied segmented images, my development benchmark reached 51.49% top-1 accuracy (156/303 queries with a matching identity in the gallery). I would like to clarify its training provenance before interpreting this as transfer to unseen animals.

The checkpoint revision is 4f1d7f2b521149e5fe34bb85f377248ce9971a7d, using the weights uploaded on 15 October 2024.

Could you help clarify:

1. Did this checkpoint's training or model selection include photographs or individuals from the FoE Czech Republic Carpathian/Bohemian collections or Šumava National Park, including material later released as CzechLynx?
2. Were any of those collections previously shared through Wildbook/Whiskerbook or directly with the MiewID team under another dataset name?
3. Is there a source list, acquisition-date cutoff, or image/identity overlap check that would help identify an independent evaluation subset?

I am distinguishing new photographs of previously seen animals from entirely unseen individuals. Any source-level clarification or recommendation for an appropriate evaluation would be very helpful.

Thank you for making these research resources available. I would be happy to share my evaluation setup and results.

Best regards,  
[Your name]  
[Affiliation, if applicable]
