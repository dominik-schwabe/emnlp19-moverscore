from moverscore import MoverScore, MoverScoreV2

data = [
    {
        "reference": "The issue isn’t about skin color, it’s about culture. Because light-skinned people have historically pushed dark-skinned people down not just in the US, or even just in European-descended cultures, look at India’s Caste system, being proud of being light skinned, especially of being a white person, includes an implication that you’re either proud of unaware of how your people have behaved throughout history. With darker-skinned people, it’s reclamation of power over themselves that they never had, but with light-skinned people the implication is that pride over your race harks back to a day when that meant you were better",
        "summary": "the issue isn't about skin color, but it's about culture, skin color skinned, skin color, skin skin, culture, india's caste system, light-skinned people have behaved throughout history.",
    },
    {
        "reference": "It's an awful description. Here's my crack at a better one So basically, twenty years before the story takes place, human settlers arrived on the planet where the story takes place, which everyone calls New World.  Unfortunately, New World was originally inhabited by an alien species called the Spackle. Of course, the humans and Spackle almost immediately got into a war, with the Spackle fighting their enemies using biological warfare. The humans eventually won, but not before the Spackle released one germ which made all humans telepathic, and another which killed all the women and most of the men. The last few hundred survivors are now stuck in one small village that's slowly dying since no one can have any kids.  Fourteen years later, Todd Hewit, the last boy to be born before all the women died, is about to officially become a man. Then, a month before Todd's birthday his parents suddenly tell him he has to run away from the village.  Turns out that even in a world where everyone's telepathic, lots of people are keeping secrets",
        "summary": "the books are pretty solid coming-of-age stories, with lots of deep stuff about war, racism and sexism, and grief. I really enjoyed them as a teen, and think they still hold up pretty well.",
    },
    {
        "reference": "The issue isn’t about skin color, it’s about culture. Because light-skinned people have historically pushed dark-skinned people down not just in the US, or even just in European-descended cultures, look at India’s Caste system, being proud of being light skinned, especially of being a white person, includes an implication that you’re either proud of unaware of how your people have behaved throughout history. With darker-skinned people, it’s reclamation of power over themselves that they never had, but with light-skinned people the implication is that pride over your race harks back to a day when that meant you were better",
        "summary": "the issue isn't about skin color, but it's about culture, skin color skinned, skin color, skin skin, culture, india's caste system, light-skinned people have behaved throughout history.",
    },
]

refs, hyps = zip(*[[value["reference"], value["summary"]] for value in data])

MoverScore.model_setup()
MoverScoreV2.model_setup()

m = MoverScore()
mv2 = MoverScoreV2()

print(m.score(refs, hyps))
print(mv2.score(refs, hyps))
