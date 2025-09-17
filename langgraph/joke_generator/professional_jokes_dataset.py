"""
Dataset of professional comedian jokes for evaluation.

This dataset contains jokes from famous comedians that can be used for
training and evaluation purposes.
"""

# Dataset of professional comedian jokes (labeled as funny)
# Source: Various famous comedians
# https://inews.co.uk/light-relief/jokes/ricky-gervais-jokes-best-golden-globes-2020-host-controversial-funniest-the-office-135797
# https://www.blackpoolgrand.co.uk/funniest-jokes-one-liners/
# https://www.vulture.com/2018/01/dave-chappelle-bird-revelation-equanimity-best-jokes.html
# https://www.scotsman.com/heritage-and-retro/heritage/billy-connollys-best-jokes-80-of-the-big-yins-funniest-jokes-and-one-liners-4458332
# https://inews.co.uk/light-relief/jokes/funny-jokes-110-funniest-best-one-liners-192413

PROFESSIONAL_JOKES = [
    {"topic": "Fishing", "joke": "Give a man a fish, and he'll probably follow you home expecting more fish.", "comedian": "Ricky Gervais"},
    {"topic": "Family", "joke": "Where there's a will – there's a relative!", "comedian": "Ricky Gervais"},
    {"topic": "Holidays", "joke": "1st of December, World Aids Day….I don't think it'll ever take off like Christmas.", "comedian": "Ricky Gervais"},
    {"topic": "Drinking", "joke": "I like a drink as much as the next man. Unless the next man is Mel Gibson.", "comedian": "Ricky Gervais"},
    {"topic": "Celebrity", "joke": "It's gonna be a night of partying and heavy drinking. Or as Charlie calls it: breakfast.", "comedian": "Ricky Gervais"},
    {"topic": "Movies", "joke": "It seems like everything this year was three-dimensional, except the characters in The Tourist.", "comedian": "Ricky Gervais"},
    {"topic": "Religion", "joke": "You won't burn in hell. But be nice anyway.", "comedian": "Ricky Gervais"},
    {"topic": "Inspiration", "joke": "My greatest hero is Nelson Mandela. What a man. Incarcerated for 25 years, he was released in 1990 and he hasn't reoffended. I think he's going straight, which shows you prison does work.", "comedian": "Ricky Gervais"},
    {"topic": "Philosophy", "joke": "Remember, when you are dead, you do not know you are dead. It is only painful for others. The same applies when you are stupid.", "comedian": "Ricky Gervais"},
    {"topic": "Life", "joke": "Mondays are fine. It's your life that sucks.", "comedian": "Ricky Gervais"},
    {"topic": "Religion", "joke": "Remember, if you don't sin, then Jesus died for nothing.", "comedian": "Ricky Gervais"},
    {"topic": "Activism", "joke": "I could solve the world's problems if I… cared.", "comedian": "Ricky Gervais"},
    {"topic": "Identity", "joke": "I can have a go at the French cause I'm half French half English with a stupid name like Gervais. No I am, I'm half French half English and um I've got qualities of both, French and English which is good, so um… I am crap in bed but at least I've got bad breath.", "comedian": "Ricky Gervais"},
    {"topic": "Military", "joke": "Do commandos not wear pants? They must wear pants, don't they?", "comedian": "Ricky Gervais"},
    {"topic": "Equality", "joke": "Same sex marriage is not a gay privilege, it's equal rights. Privilege would be something like gay people not paying taxes. Like churches don't.", "comedian": "Ricky Gervais"},
    {"topic": "Folklore", "joke": "I've never worked out what the moral of Humpty Dumpty is. I can only think of: Don't sit on a wall, if you're an egg.", "comedian": "Ricky Gervais"},
    {"topic": "Employment", "joke": "Avoid employing unlucky people – throw half of the pile of CVs in the bin without reading them.", "comedian": "Ricky Gervais"},
    {"topic": "Awards", "joke": "For any of you who don't know, the Golden Globes are just like the Oscars, but without all that esteem. The Golden Globes are to the Oscars what Kim Kardashian is to Kate Middleton. A bit louder, a bit trashier, a bit drunker, and more easily bought.", "comedian": "Ricky Gervais"},
    {"topic": "Workplace", "joke": "If your boss is getting you down, look at him through the prongs of a fork and imagine him in jail.", "comedian": "Ricky Gervais"},
    {"topic": "Humor", "joke": "I can't find someone funny whom I don't like. Hitler told great jokes.", "comedian": "Ricky Gervais"},
    {"topic": "Culture", "joke": "America champions the underdog. We champion the under dog until he's not the underdog anymore, and he annoys us.", "comedian": "Ricky Gervais"},
    {"topic": "Betrayal", "joke": "You have to be 100% behind someone, before you can stab them in the back.", "comedian": "Ricky Gervais"},
    {"topic": "Health", "joke": "Remember, being healthy is basically dying as slowly as possible.", "comedian": "Ricky Gervais"},
    {"topic": "Atheism", "joke": "I'd like to thank God for making me an atheist.", "comedian": "Ricky Gervais"},
    {"topic": "Music Industry", "joke": "Piracy doesn't kill music, boy bands do.", "comedian": "Ricky Gervais"},
    {"topic": "Wealth", "joke": "My wealth and happiness would suggest that God definitely does love me. If he existed of course. Which he doesn't.", "comedian": "Ricky Gervais"},
    {"topic": "Social Media", "joke": "Following someone on Twitter and asking them to tweet about something else is like stalking someone and asking them to go a different route.", "comedian": "Ricky Gervais"},
    {"topic": "Fame", "joke": "Please don't worship me. I'm just an ordinary guy, with lots of followers trying to spread my message. Sort of like Jesus Christ I guess.", "comedian": "Ricky Gervais"},
    {"topic": "Technology", "joke": "iPhones are Barbie Dolls for grown men. You carry them round, dress them up in little outfits, accessorise, & get a new one every year.", "comedian": "Ricky Gervais"},
    {"topic": "Generosity", "joke": "Give a man a fish, and he'll probably follow you home expecting more fish.", "comedian": "Ricky Gervais"},
    {"topic": "Environment", "joke": "It seems to be true, particularly in middle America, that those most militant about using up fossil fuels, don't actually believe in fossils", "comedian": "Ricky Gervais"},
    {"topic": "Drinking", "joke": "My father drank so heavily, when he blew on the birthday cake he lit the candles.", "comedian": "Les Dawson"},
    {"topic": "Police", "joke": "I was in my car driving back from work. A police officer pulled me over and knocked on my window. I said, 'One minute I'm on the phone.'", "comedian": "Alan Carr"},
    {"topic": "Overthinking", "joke": "I worry about ridiculous things, you know, how does a guy who drives a snowplough get to work in the morning… that can keep me awake for days.", "comedian": "Billy Connolly"},
    {"topic": "Relationships", "joke": "I used to go out with a giraffe. Used to take it to the pictures and that. You'd always get some bloke complaining that he couldn't see the screen.", "comedian": "Paul Merton"},
    {"topic": "Music", "joke": "Here's a picture of me with REM. That's me in the corner.", "comedian": "Milton Jones"},
    {"topic": "Optimism", "joke": "People say 'Bill, are you an optimist?' And I say, 'I hope so.'", "comedian": "Bill Bailey"},
    {"topic": "Customer Service", "joke": "I rang up British Telecom and said: 'I want to report a nuisance caller.' He said: 'Not you again.'", "comedian": "Tim Vine"},
    {"topic": "Obesity", "joke": "Life is like a box of chocolates. It doesn't last long if you're fat.", "comedian": "Joe Lycett"},
    {"topic": "Religion", "joke": "We weren't very religious. On Hanukkah, my mother had our menorah on a dimmer.", "comedian": "Richard Lewis"},
    {"topic": "Beauty", "joke": "My girlfriend is absolutely beautiful. Body like a Greek statue – completely pale, no arms.", "comedian": "Phil Wang"},
    {"topic": "Weather", "joke": "Normally you have news, weather and travel. But not on snow day. On a snow day, the news is weather is travel.", "comedian": "Michael McIntyre"},
    {"topic": "Personal Improvement", "joke": "I bought myself some glasses. My observational comedy improved.", "comedian": "Sara Pascoe"},
    {"topic": "Sports", "joke": "If I was an Olympic athlete, I'd rather come in last than win the silver medal. You win the gold, you feel good. You win the bronze, you think, 'at least I got something.' But you win that silver, that's like, 'Congratulations, you almost won! Of all the losers, you came in first! You're the number one loser! No one lost ahead of you!'", "comedian": "Jerry Seinfeld"},
    {"topic": "Identity", "joke": "My star sign is Pyrex. I was a test-tube baby.", "comedian": "Billy Connolly"},
    {"topic": "Marriage", "joke": "I always take my wife morning tea in my pyjamas. But is she grateful? No, she says she'd rather have it in a cup.", "comedian": "Eric Morecambe"},
    {"topic": "Shopping", "joke": "A man walks into a chemist's and says, 'Can I have a bar of soap, please?' The chemist says, 'Do you want it scented?' And the man says, 'No, I'll take it with me now.'", "comedian": "Ronnie Barker"},
    {"topic": "Crime", "joke": "Crime in multi-storey car parks. That is wrong on so many different levels.", "comedian": "Tim Vine"},
    {"topic": "Social Class", "joke": "You know you're working class when your TV is bigger than your bookcase.", "comedian": "Rob Beckett"},
    {"topic": "Animals", "joke": "Owls haven't got necks, have they? An owl is essentially a one-piece unit.", "comedian": "Ross Noble"},
    {"topic": "Fashion", "joke": "If you arrive fashionably late in Crocs, you're just late.", "comedian": "Joel Dommett"},
    {"topic": "Technology", "joke": "My phone will ring at 2am and my wife'll look at me and go, \"Who's that calling at this time?\" I say, \"I don't know. If I knew that we wouldn't need the bloody phone.\"", "comedian": "Lee Evans"},
    {"topic": "Philosophy", "joke": "I doubt there's a heaven; I think the people from hell have probably bought it for a timeshare.", "comedian": "Victoria Wood"},
    {"topic": "Fitness", "joke": "I said to the gym instructor: \"Can you teach me to do the splits?\", He said: \"How flexible are you?\", I said: \"I can't make Tuesdays.\"", "comedian": "Tommy Cooper"},
    {"topic": "Insurance", "joke": "Do Transformers get car, or life insurance?", "comedian": "Russell Howard"},
    {"topic": "Police", "joke": "Alright lads, a giant fly is attacking the police station. I've called the SWAT team!", "comedian": "Greg Davies"},
    {"topic": "Healthcare", "joke": "A good rule to remember for life is that when it comes to plastic surgery and sushi, never be attracted by a bargain.", "comedian": "Graham Norton"},
    {"topic": "Animals", "joke": "Two monkeys were getting into the bath. One said: 'Oo, oo, oo, aah aah aah.' The other replied: 'Well, put some cold in it then.'", "comedian": "Harry Hill"},
    {"topic": "Suburban Life", "joke": "My parents did just well enough so I could grow up poor around white people. When Nas and them used to talk about the projects, I used to get jealous. It sounded fun. Everybody in the projects was poor, and that's fair. But if you were poor in Silver Spring, nigga, it felt like it was only happening to you.", "comedian": "Dave Chappelle"},
    {"topic": "Cultural Identity", "joke": "What is Rachel willing to do, so that we blacks believe that she believes she is actually one of us? Bitch, are you willing to put a lien on your house so that you can invest in a mixtape that probably won't work out?", "comedian": "Dave Chappelle"},
    {"topic": "Aging", "joke": "I don't like looking at my dick anymore. My dick looks distinguished. It's old, an old-looking dick. It's got salt-and-pepper hair all around it. My dick looks like Morgan Freeman in the '90s.", "comedian": "Dave Chappelle"},
    {"topic": "Fatherhood", "joke": "This motherfucker calls me up in the middle of the night. It was one o'clock in the morning and he goes, 'Dad, don't be mad […] I'm at a party and my designated driver had too much to drink. Me and friends need you to come pick us up.' I said, 'Jesus Christ, it's one o'clock in the morning. Nigga, I am shit-faced!'", "comedian": "Dave Chappelle"},
    {"topic": "Political Commentary", "joke": "Eight years later, I'm pulling up to the polls again. This time, I'm driving a brand-new Porsche because the Obama years were very good to me […] I walked up and saw a long, long line of dusty white people […] I stood with them in line, like all us Americans are required to do in a democracy. Nobody skips the line to vote. And I listened to them say naïve, poor white people things.", "comedian": "Dave Chappelle"},
    {"topic": "Leadership", "joke": "This motherfucker [Donald Trump] grabbed the podium and he goes, 'You don't know how scary the things I read in my briefings are.' Holy shit, man, you ain't supposed to tell us that, bro!", "comedian": "Dave Chappelle"},
    {"topic": "Religious Satire", "joke": "I respect everybody's beliefs, except Amish people. They are the only ones I can say clearly, 'Their God is wrong.' The speed limit is 75 miles an hour in Ohio, and one lane of traffic is blocked by a goddamned horse and buggy?", "comedian": "Dave Chappelle"},
    {"topic": "Hollywood", "joke": "You think I go to a Hollywood meeting with all them white people by myself? I bring my nigga Mac Mittens from the streets […] He's not even qualified to listen to these meetings, he just makes me feel good.", "comedian": "Dave Chappelle"},
    {"topic": "Comedy Culture", "joke": "The tough part of being a comedian and knowing the motherfucker is, everybody comes up to me like, 'Did you know? Did you know what Louis was doing?' No, bitch, I did not know.", "comedian": "Dave Chappelle"},
    {"topic": "National Identity", "joke": "I could kill every white person in America at one time. You know how I'd do it? Just wait for the Super Bowl, and right when they sing the National Anthem, I'd have O.J. Simpson walk to the 50-yard line with them bad knees.", "comedian": "Dave Chappelle"},
    {"topic": "Gender Relations", "joke": "I used to do shows for drug dealers that wanted to clean their money up. One time I did a real good set, and these motherfuckers called me into the back room. They gave me $25,000 in cash […] I jumped on the subway and started heading towards Brooklyn at one o'clock in the morning.", "comedian": "Dave Chappelle"},
    {"topic": "Scottish Heritage", "joke": "Scottish-Americans tell you that if you want to identify tartans, it's easy – you simply look under the kilt, and if it's a quarter-pounder, you know it's a McDonald's.", "comedian": "Billy Connolly"},
    {"topic": "Judgement", "joke": "Before you judge a man, walk a mile in his shoes. After that who cares? He's a mile away and you've got his shoes!", "comedian": "Billy Connolly"},
    {"topic": "Weather", "joke": "I hate all those weathermen, too, who tell you that rain is bad weather. There's no such thing as bad weather, just the wrong clothing, so get yourself a sexy raincoat and live a little.", "comedian": "Billy Connolly"},
    {"topic": "Film Industry", "joke": "I'm a huge film star, but you have to hurry to the movies because I usually die in the first 15 f***ing minutes. I'm the only guy I know who died in a f***ing Muppet Movie.", "comedian": "Billy Connolly"},
    {"topic": "Appearance", "joke": "I always look skint. When I buy a Big Issue, people take it out of my hand and give me a pound.", "comedian": "Billy Connolly"},
    {"topic": "Sex Therapy", "joke": "One sex therapist claims that the most effective way to arouse your man is to spend 10 minutes licking his ears. Personally, I think its bollocks.", "comedian": "Billy Connolly"},
    {"topic": "Cinema", "joke": "When people say while watching a film 'did you see that? No tosser, I paid ten quid to come to the cinema and stare at the f***ing floor.", "comedian": "Billy Connolly"},
    {"topic": "Aeroplane Comfort", "joke": "I get claustrophobic easily and I don't get why aeroplane toilets don't f***ing have windows. I mean it's not as if anyone can f***ing see in. Unless of course you are the most determined pervert in the world.", "comedian": "Billy Connolly"},
    {"topic": "Astrology", "joke": "My star sign is Pyrex. I was a test-tube baby.", "comedian": "Billy Connolly"},
    {"topic": "Parenting", "joke": "Don't buy one of those baby intercoms. Babies pretend to be dead. They're bastards, and they do it on purpose.", "comedian": "Billy Connolly"},
    {"topic": "Common Sayings", "joke": "Why do people say 'Oh you want to have your cake and eat it too?' Dead right! What good is a cake if you can't eat it?", "comedian": "Billy Connolly"},
    {"topic": "Life Perception", "joke": "When people say 'life is short'. What the f***? Life is the longest damn thing anyone ever f***ing does! What can you do that's longer?", "comedian": "Billy Connolly"},
    {"topic": "Dating", "joke": "I like a woman with a head on her shoulders. I hate necks.", "comedian": "Steve Martin"},
    {"topic": "Growing Up", "joke": "I have a lot of growing up to do. I realised that the other day inside my fort.", "comedian": "Zach Galifianakis"},
    {"topic": "Employment", "joke": "I used to work at McDonald's making minimum wage. You know what that means when someone pays you minimum wage? You know what your boss was trying to say? 'Hey, if I could pay you less, I would, but it's against the law.'", "comedian": "Chris Rock"},
    {"topic": "Love", "joke": "Love is like a fart. If you have to force it it's probably s***.", "comedian": "Stephen K. Amos"},
    {"topic": "Convenience", "joke": "I like an escalator because an escalator can never break. It can only become stairs. There would never be an 'Escalator Temporarily Out of Order' sign, only 'Escalator Temporarily Stairs'.", "comedian": "Mitch Hedberg"},
    {"topic": "Sports", "joke": "If I was an Olympic athlete, I'd rather come in last than win the silver medal. You win the gold, you feel good. You win the bronze, you think, 'at least I got something.' But you win that silver, that's like, 'Congratulations, you almost won! Of all the losers, you came in first! You're the number one loser! No one lost ahead of you!'", "comedian": "Jerry Seinfeld"},
    {"topic": "Religion", "joke": "We weren't very religious. On Hanukkah, my mother had our menorah on a dimmer.", "comedian": "Richard Lewis"},
    {"topic": "Beauty", "joke": "My girlfriend is absolutely beautiful. Body like a Greek statue – completely pale, no arms.", "comedian": "Phil Wang"},
    {"topic": "Creation", "joke": "If God had written the Bible, the first line should have been 'It's round.'", "comedian": "Eddie Izzard"},
    {"topic": "Self-Improvement", "joke": "I bought myself some glasses. My observational comedy improved.", "comedian": "Sara Pascoe"},
    {"topic": "Politics", "joke": "Trump's nothing like Hitler. There's no way he could write a book.", "comedian": "Frankie Boyle"},
    {"topic": "Social Class", "joke": "You know you're working class when your TV is bigger than your book case.", "comedian": "Rob Beckett"},
    {"topic": "Conflict", "joke": "Most of my life is spent avoiding conflict. I hardly ever visit Syria.", "comedian": "Alex Horne"},
    {"topic": "Relaxation", "joke": "A spa hotel? It's like a normal hotel, only in reception there's a picture of a pebble.", "comedian": "Rhod Gilbert"},
    {"topic": "Health", "joke": "Life is like a box of chocolates. It doesn't last long if you're fat.", "comedian": "Joe Lycett"},
    {"topic": "Career", "joke": "My Dad said, always leave them wanting more. Ironically, that's how he lost his job in disaster relief.", "comedian": "Mark Watson"},
    {"topic": "Memory", "joke": "Apparently smoking cannabis can affect your short term memory. Well if that's true, what do you think smoking cannabis does?", "comedian": "Mickey P Kerr"},
    {"topic": "Philosophy", "joke": "How many philosophers does it take to change a lightbulb?…. none. They're not really into that sort of thing. If it's that dark, light a candle.", "comedian": "Phil Cornwell"},
    {"topic": "Marriage", "joke": "The first time I met my wife, I knew she was a keeper. She was wearing massive gloves.", "comedian": "Alun Cochrane"},
    {"topic": "Childhood", "joke": "As a kid I was made to walk the plank. We couldn't afford a dog.", "comedian": "Gary Delaney"},
    {"topic": "Misunderstanding", "joke": "Two fish in a tank. One says: 'How do you drive this thing?'", "comedian": "Peter Kay"},
    {"topic": "Entertainment", "joke": "I saw a documentary on how ships are kept together. Riveting!", "comedian": "Stewart Francis"},
    {"topic": "Music", "joke": "People who like trance music are very persistent. They don't techno for an answer.", "comedian": "Joel Dommett"},
    {"topic": "Dating", "joke": "I used to go out with a giraffe. Used to take it to the pictures and that. You'd always get some bloke complaining that he couldn't see the screen. It's a giraffe, mate. What do you expect? 'Well he can take his hat off for a start!'", "comedian": "Paul Merton"},
    {"topic": "Weather", "joke": "Normally you have news, weather and travel. But not on snow day. On a snow day, news is weather is travel.", "comedian": "Michael McIntyre"},
    {"topic": "Music", "joke": "Here's a picture of me with REM. That's me in the corner.", "comedian": "Milton Jones"},
    {"topic": "Sarcasm", "joke": "Someone showed me a photograph of my local MP the other day. 'Would you buy a second-hand car from this man?' they asked. 'Would you buy a second-hand car?' I replied.", "comedian": "Miles Jupp"},
    {"topic": "Culture", "joke": "With stand-up in Britain, what you have to do is bloody swearing. In Germany, we don't have to swear. Reason being, things work.", "comedian": "Henning When"},
    {"topic": "Learning", "joke": "I'm learning the hokey cokey. Not all of it. But – I've got the ins and outs.", "comedian": "Iain Stirling"},
    {"topic": "Identity", "joke": "Roses are red, violets are blue, I'm a schizophrenic, and so am I.", "comedian": "Billy Connolly"},
    {"topic": "Parenting", "joke": "My mother told me, you don't have to put anything in your mouth you don't want to. Then she made me eat broccoli, which felt like double standards.", "comedian": "Sarah Millican"},
    {"topic": "Vengeance", "joke": "My therapist says I have a preoccupation with vengeance. We'll see about that.", "comedian": "Stewart Francis"},
    {"topic": "Family", "joke": "I'm sure wherever my Dad is, he's looking down on us. He's not dead, just very condescending.", "comedian": "Jack Whitehall"},
    {"topic": "Marriage", "joke": "'What's a couple?' I asked my mum. She said, 'Two or three'. Which probably explains why her marriage collapsed.", "comedian": "Josie Long"},
    {"topic": "Injury", "joke": "The easiest time to add insult to injury is when you're signing somebody's cast.", "comedian": "Demetri Martin"},
    {"topic": "Communication", "joke": "I was in my car driving back from work. A police officer pulled me over and knocked on my window. I said, 'One minute I'm on the phone.'", "comedian": "Alan Carr"},
    {"topic": "Afterlife", "joke": "I doubt there's a heaven; I think the people from hell have probably bought it for a timeshare.", "comedian": "Victoria Wood"},
    {"topic": "Flexibility", "joke": "I said to the gym instructor: 'Can you teach me to do the splits?' He said: 'How flexible are you?' I said: 'I can't make Tuesdays.'", "comedian": "Tommy Cooper"},
    {"topic": "Misunderstanding", "joke": "A man walks into a chemist's and says, 'Can I have a bar of soap, please?' The chemist says, 'Do you want it scented?' And the man says, 'No, I'll take it with me now.'", "comedian": "Ronnie Barker"},
    {"topic": "Humor", "joke": "It's really hard to define 'virtue signalling', as I was saying the other day to some of my Muslim friends over a fair-trade coffee in our local feminist bookshop.", "comedian": "Lucy Porter"},
    {"topic": "Creation", "joke": "If we were truly created by God, then why do we still occasionally bite the insides of our own mouths?", "comedian": "Dara Ó Briain"},
    {"topic": "Insurance", "joke": "Do Transformers get car, or life insurance?", "comedian": "Russell Howard"},
    {"topic": "Emergency", "joke": "Alright lads, a giant fly is attacking the police station. I've called the SWAT team!", "comedian": "Greg Davies"},
    {"topic": "Consumerism", "joke": "A good rule to remember for life is that when it comes to plastic surgery and sushi, never be attracted by a bargain.", "comedian": "Graham Norton"},
    {"topic": "Family", "joke": "My father drank so heavily, when he blew on the birthday cake he lit the candles.", "comedian": "Les Dawson"},
    {"topic": "Therapy", "joke": "I've been feeling suicidal so my therapist suggested I do CBT. Now I can ride a motorbike, how's that going to help?", "comedian": "Eric Lampaert"},
]


def get_jokes_by_topic(topic: str):
    """Get all jokes for a specific topic."""
    return [joke for joke in PROFESSIONAL_JOKES if joke["topic"].lower() == topic.lower()]


def get_jokes_by_comedian(comedian: str):
    """Get all jokes by a specific comedian."""
    return [joke for joke in PROFESSIONAL_JOKES if comedian.lower() in joke["comedian"].lower()]


def get_random_joke():
    """Get a random joke from the dataset."""
    import random
    return random.choice(PROFESSIONAL_JOKES)


def get_all_topics():
    """Get all unique topics in the dataset."""
    return list(set(joke["topic"] for joke in PROFESSIONAL_JOKES))


def get_all_comedians():
    """Get all unique comedians in the dataset."""
    return list(set(joke["comedian"] for joke in PROFESSIONAL_JOKES))