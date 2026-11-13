# Real-world benchmark (journal 18)

Seven openly licensed ASL videos with English text, seven signers, used to score the **whole translator** (detector,
letter model, decoder, word spotting) on real footage. The answer key was written and committed **before any model
was run on these videos** (`SHA256SUMS`, and the git history).

| Video (Wikimedia Commons) | Signer | English text | Split |
|---|---|---|---|
| [(ASL) - What is E. coli](https://commons.wikimedia.org/wiki/File:(ASL)_-_What_is_E._coli.webm) (USDA FSIS) | interpreter in a kitchen; USDA presenter in intro/outro (also seen in the Campylobacter video) | burned-in captions | test |
| [(ASL) Microwaving Convenience Foods](https://commons.wikimedia.org/wiki/File:(ASL)_Microwaving_Convenience_Foods_USDA.webm) (USDA FSIS) | interpreter | burned-in captions | tune |
| [FEMA Accessible Hurricane Harvey - Shelter in Place](https://commons.wikimedia.org/wiki/File:FEMA_Accessible_Hurricane_Harvey_-_Shelter_in_Place.webm) | FEMA interpreter | subtitle track | test |
| [This Is A Tour Of The West Wing (In Sign Language)](https://commons.wikimedia.org/wiki/File:This_Is_A_Tour_Of_The_West_Wing_(In_Sign_Language).webm) (White House) | Leah Katz-Hernandez, Deaf native signer, full-body shots | subtitle track | test |
| [Protective Actions for Radiation Emergencies - Get Inside...](https://commons.wikimedia.org/wiki/File:Protective_Actions_for_Radiation_Emergencies_-_Get_Inside,_Stay_Inside,_Stay_Tuned_(ASL).webm) (CDC) | interpreter, small inset | burned-in captions | test |
| [Protective Actions for Radiation Emergencies - Self-Decontamination](https://commons.wikimedia.org/wiki/File:Protective_Actions_for_Radiation_Emergencies_-_Self-Decontamination_(ASL).webm) (CDC) | interpreter, small inset | burned-in captions | test |
| [Lightning Safety for the Deaf and Hard of Hearing Community](https://commons.wikimedia.org/wiki/File:Lightning_Safety_for_the_Deaf_and_Hard_of_Hearing_Community.webm) (NWS) | signer | on-screen summary text | tune |

All are US federal government works (public domain); the subtitle tracks are from Wikimedia Commons TimedText.
The videos themselves are not in the repository (`clips/benchmark/`, git-ignored); `benchmark/videos.txt` lists their download URLs (Wikimedia asks for a descriptive User-Agent).

**Files.** `<video>.tsv`: the English text with start times (burned-in captions read from frames, ±1 s; or the
subtitle track). `key.json`: terms likely to be fingerspelled (names, acronyms, technical terms without a common sign,
chosen from the English text), each checked on the video and marked `spelled` / `unclear` / `signed` / `none`.
"spelled" means fingerspelling is visible within [t-0.5, t+5.5] s; which exact word was spelled can't always be read at
that resolution. Four other CDC/COVID videos were downloaded but have no English text and are not used.
