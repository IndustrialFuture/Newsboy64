from benchmark import CustomQuestion

ds = [
        CustomQuestion(
            title="Will the price of one bitcoin on July 1 2025 be more than USD $120,000?",
            resolution_criteria="No additional criterion.",
            description="The second 2025 bitcoin surge has pushed bitcoin prices above $100,000 again, but it remains to be seen whether this will continue rising.",
            fine_print="The price units are in USD, and the resolution source is Yahoo Finance",
            community_prediction=0.40,
            url="https://www.gjopen.com/questions/4162-what-will-be-the-average-us-city-price-of-a-pound-of-ground-beef-in-august-2025",
            type = "binary",
        ),
        CustomQuestion(
            title="What will be the average US city price of a pound of ground beef in August 2025?",
            resolution_criteria="No additional criterion.",
            description="The US beef market, already facing high demand and tight supplies, is expected to be hit hard by tariff policies with Canada and Mexico (Economist, Drovers). The question will be suspended on 31 August 2025 and the outcome determined using Bureau of Labor Statistics (BLS) data as reported by the Federal Reserve Economic Data database (FRED) (FRED).",
            fine_print="No additional fine print.",
            community_prediction=[0.01, 0.02, 0.06, 0.19, 0.30, 0.21, 0.12, 0.04, 0.05],
            url="https://www.gjopen.com/questions/4162-what-will-be-the-average-us-city-price-of-a-pound-of-ground-beef-in-august-2025",
            type = "multiple_choice",
            options = ["Less than $5.000", "At least $5.000, but less than $5.250", "At least $5.250, but less than $5.500", "At least $5.500, but less than $5.750", "At least $5.750, but less than $6.000", "At least $6.000, but less than $6.250", "At least $6.250, but less than $6.500", "At least $6.500, but less than $6.750", "$6.750 or more"]
        ),
        CustomQuestion(
            title="Will François Bayrou cease to be the prime minister of France before 1 October 2025?",
            resolution_criteria="No additional criterion.",
            description="After early elections led to a sharply divided National Assembly in 2024, the current French government is struggling to survive (Economist, MSN [AFP], France 24).",
            fine_print="No additional fine print.",
            community_prediction=0.2754,
            url="https://www.gjopen.com/questions/4080-will-francois-bayrou-cease-to-be-the-prime-minister-of-france-before-1-october-2025/crowd_forecast",
            type = "binary"
        ),
        
        CustomQuestion(
            title="Will Elon Musk cease to be an advisor to Donald Trump and face public criticism from Donald Trump before 2026?",
            resolution_criteria="This question will resolve as Yes if, before January 1, 2026, credible sources have reported both of the following: Elon Musk no longer serves an advisory role to Donald Trump. This will be considered to be true if Elon Musk ceases to hold either a formal federal government position or an advisory position serving at an organization governed by the Federal Advisory Committee Act (FACA) or any other officially recognized advisory organization. This includes if Elon Musk does not at any point attain such a position during the year 2025. If Elon Musk ceases to hold an advisory position because he is becoming a full federal government employee, or vice versa, this will not be considered to have occurred. Musk must cease to be both a government employee and an advisor simultaneously. Transfers in status where Musk temporarily ceases to hold either role do not count. Donald Trump has publicly criticized Elon Musk during a time while Musk is no longer serving on a government or advisory position. Public criticism will generally not be taken to include jokes, backhanded compliments, snide remarks, or other non-serious insults or veiled insults. To count, a criticism must unambiguously convey dislike or dissatisfaction with Musk. Otherwise, the question will resolve as No.",
            description="""
            President Elect Donald Trump has announced that Elon Musk will serve an advisory role as head of the planned Department of Government Efficiency (DOGE), alongside Vivek Ramaswamy. Trump has said that DOGE will ". . .dismantle Government Bureaucracy, slash excess regulations, cut wasteful expenditures, and restructure Federal Agencies. . ." DOGE is anticipated to be structured as an advisory commission, because DOGE does not yet exist and because federal government offices must be created by Congress. The Federal Advisory Committee Act (FACA) typically governs such advisory committees.
            """,
            fine_print="""
            Metaculus will consult media characterizations of Trump's comments on Musk to aid in determining if criticism has occurred. Metaculus will assess multiple sources of various political leanings to aid in limiting bias in assessing whether Trump has publicly criticized Musk.
            A criticism must be public to count, indirect reports, such as reporting on private conversations or statements that occurred behind closed doors, will not count.
            Whether Musk is fired or resigns is immaterial.
            If the outcome is disputed, a panel of three Admins will vote on how the question should resolve. """,
            community_prediction=0.60,
            url="https://www.metaculus.com/questions/30891/elon-musk-and-donald-trump-falling-out-before-2026/",
            type = "binary"
        )
        ,

        CustomQuestion(
        title="Will Volodymyr Zelensky either flee Ukraine or cease to be its president before 1 January 2026?",
        description="As talks between the US and Russia continue without Ukraine about the future of the conflict, President Zelensky has stated he is ready to resign in exchange for Ukraine’s membership in NATO (Politico, CNN, CBS News) [oai_citation_attribution:0‡gjopen.com](https://www.gjopen.com/questions/4118-will-volodymyr-zelensky-either-flee-ukraine-or-cease-to-be-its-president-before-1-january-2026#:~:text=As%20talks%20between%20the%20US,of%20Ukraine%20by%20the%20European).",
        resolution_criteria="Resolves YES if, before January 1, 2026, Zelensky either flees Ukraine or ceases to be the president of Ukraine. The expiration of Zelensky’s term of office does not count as 'ceasing' so long as it is extended by law, and unrecognized claims of succession while Zelensky still claims the presidency will not count. Whether Zelensky has fled Ukraine will be determined via credible media reports [oai_citation_attribution:1‡gjopen.com](https://www.gjopen.com/questions/4118-will-volodymyr-zelensky-either-flee-ukraine-or-cease-to-be-its-president-before-1-january-2026#:~:text=he%20is%20ready%20to%20resign,will%20be%20determined%20using%20credible).",
        fine_print="For the purposes of this question, the expiration of Zelensky’s term as president does not count as ceasing so long as it is extended by law [oai_citation_attribution:2‡gjopen.com](https://www.gjopen.com/questions/4118-will-volodymyr-zelensky-either-flee-ukraine-or-cease-to-be-its-president-before-1-january-2026#:~:text=News%20www,of%20Ukraine%20by%20the%20European). Claims of succession by someone not recognized as president by the EU while Zelensky asserts he is still president will not count. Fleeing Ukraine will be determined using credible media reporting [oai_citation_attribution:3‡gjopen.com](https://www.gjopen.com/questions/4118-will-volodymyr-zelensky-either-flee-ukraine-or-cease-to-be-its-president-before-1-january-2026#:~:text=extended%20by%20law%20,source%20media%20reporting).",
        community_prediction=0.2528,
        url="https://www.gjopen.com/questions/4118-will-volodymyr-zelensky-either-flee-ukraine-or-cease-to-be-its-president-before-1-january-2026",
        type="binary"
        ),
        CustomQuestion(
            title="In the current conflict in Ukraine, when will Russia and Ukraine announce a ceasefire with an intended indefinite duration or an intended duration of at least 28 days?",
            description="Despite various calls for a ceasefire in Ukraine, the fighting continues (Economist, Washington Post, AP, Newsweek) [oai_citation_attribution:5‡gjopen.com](https://www.gjopen.com/questions/3847-in-the-current-conflict-in-ukraine-when-will-russia-and-ukraine-announce-a-ceasefire-with-an-intended-indefinite-duration-or-an-intended-duration-of-at-least-28-days#:~:text=Despite%20various%20calls%20for%20a,date%20a%20ceasefire%20will%20take).",
            resolution_criteria="Resolves to the option corresponding to the date range in which Russia and Ukraine announce a qualifying ceasefire. A qualifying ceasefire must be acknowledged by both Russia and Ukraine, cover the whole of Ukraine’s territory, and have an intended duration of at least 28 days or be indefinite [oai_citation_attribution:6‡gjopen.com](https://www.gjopen.com/questions/3847-in-the-current-conflict-in-ukraine-when-will-russia-and-ukraine-announce-a-ceasefire-with-an-intended-indefinite-duration-or-an-intended-duration-of-at-least-28-days#:~:text=%28Economist%2C%20MSN%20,a%20ceasefire%20will%20be%20determined).",
            fine_print="The Ukrainian government (as recognized by the EU) must be party to the announced ceasefire, and 'Ukraine' includes all internationally recognized territory (including Donbas and Crimea) [oai_citation_attribution:7‡gjopen.com](https://www.gjopen.com/questions/3847-in-the-current-conflict-in-ukraine-when-will-russia-and-ukraine-announce-a-ceasefire-with-an-intended-indefinite-duration-or-an-intended-duration-of-at-least-28-days#:~:text=be%20acknowledged%20by%20both%20Russia,ceasefire%20that%20is%20extended%20for). The date a ceasefire takes effect is immaterial. An intended duration qualifies only if agreed upon at inception (e.g., extending a shorter ceasefire does not count) [oai_citation_attribution:8‡gjopen.com](https://www.gjopen.com/questions/3847-in-the-current-conflict-in-ukraine-when-will-russia-and-ukraine-announce-a-ceasefire-with-an-intended-indefinite-duration-or-an-intended-duration-of-at-least-28-days#:~:text=be%20party%20to%20the%20announced,its%20intended%20duration%20is%20immaterial). Whether the ceasefire holds for its intended duration is not relevant.",
            community_prediction=[0, 0.17, 0.26, 0.57],
            url="https://www.gjopen.com/questions/3847-in-the-current-conflict-in-ukraine-when-will-russia-and-ukraine-announce-a-ceasefire-with-an-intended-indefinite-duration-or-an-intended-duration-of-at-least-28-days",
            type="multiple_choice",
            options=["Before 20 January 2025", "Between 20 January 2025 and 25 May 2025", "Between 26 May 2025 and 28 September 2025", "Not before 29 September 2025"]
        ),
        CustomQuestion(
            title="What will be the closing market capitalization for NVIDIA Corp on 30 September 2025?",
            description="NVIDIA has grown to dominate the semiconductor industry as the race for AI continues (Economist, Elektor Magazine, Investopedia) [oai_citation_attribution:10‡gjopen.com](https://www.gjopen.com/questions/3857-what-will-be-the-closing-market-capitalization-for-nvidia-corp-on-30-september-2025#:~:text=NVIDIA%20has%20grown%20to%20dominate,Market%20Cap).",
            resolution_criteria="The question will be suspended on 29 September 2025 and resolved using NVIDIA’s market capitalization at market close on 30 September 2025, as reported by CNBC (see 'Market Cap') [oai_citation_attribution:11‡gjopen.com](https://www.gjopen.com/questions/3857-what-will-be-the-closing-market-capitalization-for-nvidia-corp-on-30-september-2025#:~:text=continues%20%28Economist%2C%20Elektor%20Magazine%2C%20Investopedia%29,Market%20Cap). The outcome will correspond to the range that the market cap falls into.",
            fine_print="Market capitalization will be taken from CNBC’s data on that date. All values are in US dollars, in trillions. (E.g., '$3.0T' = $3.0 trillion.)",
            community_prediction=[0.17, 0.21, 0.3, 0.2, 0.08, 0.03, 0.01, 0.0, 0.0],
            url="https://www.gjopen.com/questions/3857-what-will-be-the-closing-market-capitalization-for-nvidia-corp-on-30-september-2025",
            type="multiple_choice",
            options=["Less than $2.5 trillion", "At least $2.5 trillion, but less than $3.0 trillion", "At least $3.0 trillion, but less than $3.5 trillion", "At least $3.5 trillion, but less than $4.0 trillion", "At least $4.0 trillion, but less than $4.5 trillion", "At least $4.5 trillion, but less than $5.0 trillion", "At least $5.0 trillion, but less than $6.0 trillion", "At least $6.0 trillion, but less than $7.0 trillion", "$7.0 trillion or more"]
        ),
        CustomQuestion(
            title="How many NATO member states will spend 2.0% or more of their GDP on defense in 2025?",
            description="With Russia’s invasion of Ukraine, NATO’s role and defense spending are in the spotlight (Economist, US News, NATO) [oai_citation_attribution:13‡gjopen.com](https://www.gjopen.com/questions/3849-how-many-nato-member-states-will-spend-2-0-or-more-of-their-gdp-on-defense-in-2025#:~:text=With%20Russia%27s%20invasion%20of%20Ukraine%2C,between%20the%20PDF%20and%20the).",
            resolution_criteria="Resolves to the number of NATO countries spending at least 2.0% of their GDP on defense in 2025, as determined by NATO’s first annual report released in 2025 that includes 2025 data (including estimates) [oai_citation_attribution:14‡gjopen.com](https://www.gjopen.com/questions/3849-how-many-nato-member-states-will-spend-2-0-or-more-of-their-gdp-on-defense-in-2025#:~:text=defense%20spending%2C%20are%20in%20the,will%20be).",
            fine_print="The outcome will use data from NATO’s defense expenditure report (Table 3) released by mid-2025 [oai_citation_attribution:15‡gjopen.com](https://www.gjopen.com/questions/3849-how-many-nato-member-states-will-spend-2-0-or-more-of-their-gdp-on-defense-in-2025#:~:text=News%20%26%20World%20Report%20www,GDP%20on%20defense%2C%20up%20from). If PDF and spreadsheet figures differ, the spreadsheet values will be used [oai_citation_attribution:16‡gjopen.com](https://www.gjopen.com/questions/3849-how-many-nato-member-states-will-spend-2-0-or-more-of-their-gdp-on-defense-in-2025#:~:text=than%20mid,which%20included%20estimates%20for%202024). (According to NATO’s June 2024 report, 23 countries met the 2.0% target in 2024, up from 10 in 2023 [oai_citation_attribution:17‡gjopen.com](https://www.gjopen.com/questions/3849-how-many-nato-member-states-will-spend-2-0-or-more-of-their-gdp-on-defense-in-2025#:~:text=used,and%20therefore%20no%20defense%20expenditures).) Iceland is excluded, as it has no defense expenditures.",
            community_prediction=[0.04, 0.05, 0.28, 0.48, 0.15],
            url="https://www.gjopen.com/questions/3849-how-many-nato-member-states-will-spend-2-0-or-more-of-their-gdp-on-defense-in-2025",
            type="multiple_choice",
            options=["19 or fewer", "20 or 21", "22 or 23", "24 or 25", "26 or more"]
        ),
        CustomQuestion(
            title="What will be the effective US tariff rate for Chinese imports for consumption in July 2025?",
            description="President-elect Trump’s plans for US tariff policy toward China have raised concerns about impacts on international trade (Economist, Bloomberg, Newsweek) [oai_citation_attribution:19‡gjopen.com](https://www.gjopen.com/questions/3854-what-will-be-the-effective-us-tariff-rate-for-chinese-imports-for-consumption-in-july-2025#:~:text=President,suspended%20on%2031%20July%202025).",
            resolution_criteria="The question will be suspended on 31 July 2025 and resolved using data from the US Census Bureau’s USA Trade Online. The effective tariff rate for July 2025 will be calculated as (Calculated Duty ÷ Customs Value) × 100%, using the values for imports from China [oai_citation_attribution:20‡gjopen.com](https://www.gjopen.com/questions/3854-what-will-be-the-effective-us-tariff-rate-for-chinese-imports-for-consumption-in-july-2025#:~:text=Newsweek%29,and%20select%20the) [oai_citation_attribution:21‡gjopen.com](https://www.gjopen.com/questions/3854-what-will-be-the-effective-us-tariff-rate-for-chinese-imports-for-consumption-in-july-2025#:~:text=calculate%20the%20effective%20tariff%20rate,2024YTD%2015%20November%202024).",
            fine_print="Data will be obtained by selecting 'Customs Value (Cons)' and 'Calculated Duty' for China for July 2025 in USA Trade Online [oai_citation_attribution:22‡gjopen.com](https://www.gjopen.com/questions/3854-what-will-be-the-effective-us-tariff-rate-for-chinese-imports-for-consumption-in-july-2025#:~:text=After%20logging%20in%2C%20click%20,Export). In July 2024, the effective tariff rate was ~11.279% [oai_citation_attribution:23‡gjopen.com](https://www.gjopen.com/questions/3854-what-will-be-the-effective-us-tariff-rate-for-chinese-imports-for-consumption-in-july-2025#:~:text=calculate%20the%20effective%20tariff%20rate,2024YTD%2015%20November%202024). Free registration is required to access the data [oai_citation_attribution:24‡gjopen.com](https://www.gjopen.com/questions/3854-what-will-be-the-effective-us-tariff-rate-for-chinese-imports-for-consumption-in-july-2025#:~:text=Newsweek%29,and%20select%20the).",
            community_prediction=[0.03, 0.05, 0.05, 0.04, 0.09, 0.11, 0.63],
            url="https://www.gjopen.com/questions/3854-what-will-be-the-effective-us-tariff-rate-for-chinese-imports-for-consumption-in-july-2025",
            type="multiple_choice",
            options=["Less than 10.0%", "At least 10.0%, but less than 11.5%", "At least 11.5%, but less than 13.0%", "At least 13.0%, but less than 14.5%", "At least 14.5%, but less than 16.0%", "At least 16.0%, but less than 17.5%", "17.5% or more"]
        ),
        CustomQuestion(
            title="What will be the closing price for gold per troy ounce on 23 May 2025, according to Trading Economics?",
            description="Gold prices fell from all-time highs in the wake of U.S. elections, but demand is expected to remain strong (Economist, Reuters, Investopedia) [oai_citation_attribution:26‡gjopen.com](https://www.gjopen.com/questions/3859-what-will-be-the-closing-price-for-gold-per-troy-ounce-on-23-may-2025-according-to-trading-economics#:~:text=Gold%20prices%20fell%20from%20all,Trading%20Economics).",
            resolution_criteria="The question will be suspended on 22 May 2025 and resolved using the closing price of gold (USD per troy ounce) on 23 May 2025, as reported by Trading Economics [oai_citation_attribution:27‡gjopen.com](https://www.gjopen.com/questions/3859-what-will-be-the-closing-price-for-gold-per-troy-ounce-on-23-may-2025-according-to-trading-economics#:~:text=,Trading%20Economics). The price will be taken as reported in the U.S. version of the site (to avoid time zone discrepancies) [oai_citation_attribution:28‡gjopen.com](https://www.gjopen.com/questions/3859-what-will-be-the-closing-price-for-gold-per-troy-ounce-on-23-may-2025-according-to-trading-economics#:~:text=will%20be%20suspended%20on%2022,Trading%20Economics).",
            fine_print="If Trading Economics lists prices by date differently depending on user location, the U.S.-based date labeling will be used for consistency [oai_citation_attribution:29‡gjopen.com](https://www.gjopen.com/questions/3859-what-will-be-the-closing-price-for-gold-per-troy-ounce-on-23-may-2025-according-to-trading-economics#:~:text=will%20be%20suspended%20on%2022,Trading%20Economics).",
            community_prediction=[0.0, 0.0, 0.01, 0.01, 0.02, 0.03, 0.06, 0.11, 0.19, 0.20, 0.18, 0.19],
            url="https://www.gjopen.com/questions/3859-what-will-be-the-closing-price-for-gold-per-troy-ounce-on-23-may-2025-according-to-trading-economics",
            type="multiple_choice",
            options=["Less than $2,300", "$2,300 to <$2,400", "$2,400 to <$2,500", "$2,500 to <$2,600", "$2,600 to <$2,700", "$2,700 to <$2,800", "$2,800 to <$2,900", "$2,900 to <$3,000", "$3,000 to <$3,100", "$3,100 to <$3,200", "$3,200 to <$3,300", "$3,300 or more"]
        ),
        CustomQuestion(
            title="Will Min Aung Hlaing or his military successor(s) cease to be the head of government of Myanmar before 1 October 2025?",
            description="Myanmar’s ruling military government (Tatmadaw), led by Prime Minister Min Aung Hlaing since a February 2021 coup, has faced increased fighting with armed opposition groups (Economist, US News, AP, Britannica) [oai_citation_attribution:32‡gjopen.com](https://www.gjopen.com/questions/3848-will-min-aung-hlaing-or-his-military-successor-s-cease-to-be-the-head-of-government-of-myanmar-before-1-october-2025#:~:text=The%20ruling%20military%20,or%20his%20military).",
            resolution_criteria="Resolves YES if Min Aung Hlaing, or any military successor who replaces him as head of government, is no longer the head of government of Myanmar before 1 October 2025. This includes being removed from power or fleeing Myanmar while in power [oai_citation_attribution:33‡gjopen.com](https://www.gjopen.com/questions/3848-will-min-aung-hlaing-or-his-military-successor-s-cease-to-be-the-head-of-government-of-myanmar-before-1-october-2025#:~:text=AP%2C%20Britannica%20,source%20reporting). Otherwise, resolves NO.",
            fine_print="If the junta names a new leader, that means Min Aung Hlaing ceased to be head of government. If Min Aung Hlaing (or his successor) flees Myanmar, that counts as ceasing to be head of government [oai_citation_attribution:34‡gjopen.com](https://www.gjopen.com/questions/3848-will-min-aung-hlaing-or-his-military-successor-s-cease-to-be-the-head-of-government-of-myanmar-before-1-october-2025#:~:text=Aung%20Hlaing%20as%20head%20of,source%20reporting). Determination will rely on credible media reports.",
            community_prediction=0.1851,
            url="https://www.gjopen.com/questions/3848-will-min-aung-hlaing-or-his-military-successor-s-cease-to-be-the-head-of-government-of-myanmar-before-1-october-2025",
            type="binary"
        ),
        CustomQuestion(
            title="What will be the monthly global land and sea temperature anomaly for July 2025, according to NOAA?",
            description="Recorded global surface temperatures have risen steadily for decades (Economist, NASA) [oai_citation_attribution:36‡gjopen.com](https://www.gjopen.com/questions/3855-what-will-be-the-monthly-global-land-and-sea-temperature-anomaly-for-july-2025-according-to-the-national-oceanic-and-atmospheric-administration-noaa#:~:text=Recorded%20global%20surface%20temperatures%20have,22%C2%B0C).",
            resolution_criteria="Resolves to the reported global land-and-ocean surface temperature anomaly for July 2025 (relative to the 20th-century average), as published by NOAA’s National Centers for Environmental Information. The value will be taken from NOAA’s data release in August 2025 [oai_citation_attribution:37‡gjopen.com](https://www.gjopen.com/questions/3855-what-will-be-the-monthly-global-land-and-sea-temperature-anomaly-for-july-2025-according-to-the-national-oceanic-and-atmospheric-administration-noaa#:~:text=%28Economist%2C%20NASA%29,22%C2%B0C).",
            fine_print="The question will be suspended on 31 July 2025, and the NOAA data (monthly global land & ocean anomaly) expected in August 2025 will be used [oai_citation_attribution:38‡gjopen.com](https://www.gjopen.com/questions/3855-what-will-be-the-monthly-global-land-and-sea-temperature-anomaly-for-july-2025-according-to-the-national-oceanic-and-atmospheric-administration-noaa#:~:text=%28Economist%2C%20NASA%29,22%C2%B0C). As of launch, July 2024’s anomaly was 1.22°C [oai_citation_attribution:39‡gjopen.com](https://www.gjopen.com/questions/3855-what-will-be-the-monthly-global-land-and-sea-temperature-anomaly-for-july-2025-according-to-the-national-oceanic-and-atmospheric-administration-noaa#:~:text=published%20by%20NOAA%2C%20expected%20in,22%C2%B0C).",
            community_prediction=[0.0245, 0.048, 0.4166, 0.3653, 0.1088],
            url="https://www.gjopen.com/questions/3855-what-will-be-the-monthly-global-land-and-sea-temperature-anomaly-for-july-2025-according-to-the-national-oceanic-and-atmospheric-administration-noaa",
            type="multiple_choice",
            options=["Less than 0.90°C", "0.90°C to <1.10°C", "1.10°C to <1.30°C", "1.30°C to <1.50°C", "1.50°C or more"]
        ),
        CustomQuestion(
            title="Will the presidents of Russia and the United States meet in person before 21 July 2025?",
            description="Former U.S. President Donald Trump has claimed that Russia’s President Vladimir Putin wants a meeting regarding the war in Ukraine (CNBC, Bloomberg) [oai_citation_attribution:41‡gjopen.com](https://www.gjopen.com/questions/3964-will-the-presidents-of-russia-and-the-united-states-meet-in-person-before-21-july-2025#:~:text=Donald%20Trump%20has%20said%20that,the%20relevant%20country%20will%20count).",
            resolution_criteria="Resolves YES if Joe Biden (or a U.S. acting president) and Vladimir Putin meet face-to-face before 21 July 2025 [oai_citation_attribution:42‡gjopen.com](https://www.gjopen.com/questions/3964-will-the-presidents-of-russia-and-the-united-states-meet-in-person-before-21-july-2025#:~:text=Donald%20Trump%20has%20said%20that,the%20relevant%20country%20will%20count). The meeting can occur in any venue and does not need to be a formal bilateral summit. A meeting with an acting president (under constitutional succession) would count.",
            fine_print="Credible media reporting of an in-person meeting will be used to resolve this question. If no such meeting occurs by the deadline, it resolves NO.",
            community_prediction=0.5799,
            url="https://www.gjopen.com/questions/3964-will-the-presidents-of-russia-and-the-united-states-meet-in-person-before-21-july-2025",
            type="binary"
        ),
        CustomQuestion(
            title="What will be the closing market capitalization of Boeing on 1 August 2025?",
            description="Aircraft manufacturer Boeing has weathered numerous challenges, including product problems and a major labor strike (Economist, KIRO7 News, MSN) [oai_citation_attribution:46‡gjopen.com](https://www.gjopen.com/questions/3886-what-will-be-the-closing-market-capitalization-of-boeing-on-1-august-2025#:~:text=Aircraft%20manufacturer%20Boeing%20has%20weathered,MARKET%20CAP).",
            resolution_criteria="The question will be suspended on 31 July 2025 and resolved using Boeing’s market capitalization at market close on 1 August 2025, as reported by an official source (e.g. Google Finance, see 'Market Cap') [oai_citation_attribution:47‡gjopen.com](https://www.gjopen.com/questions/3886-what-will-be-the-closing-market-capitalization-of-boeing-on-1-august-2025#:~:text=product%20problems%20and%20a%20major,MARKET%20CAP). The result will correspond to the range in which the market cap falls.",
            fine_print="Market cap values are in U.S. dollars. If multiple sources exist, official financial data (e.g. Google Finance) will be used for consistency [oai_citation_attribution:48‡gjopen.com](https://www.gjopen.com/questions/3886-what-will-be-the-closing-market-capitalization-of-boeing-on-1-august-2025#:~:text=product%20problems%20and%20a%20major,MARKET%20CAP).",
            community_prediction=[0.01, 0.02, 0.06, 0.13, 0.19, 0.24, 0.19, 0.1, 0.05, 0.01, 0.0, 0.0],
            url="https://www.gjopen.com/questions/3886-what-will-be-the-closing-market-capitalization-of-boeing-on-1-august-2025",
            type="multiple_choice",
            options=["Less than $60 billion", "$60B to <$75B", "$75B to <$90B", "$90B to <$105B", "$105B to <$120B", "$120B to <$135B", "$135B to <$150B", "$150B to <$165B", "$165B to <$180B", "$180B to <$195B", "$195B to <$210B", "$210B or more"]
        ),
        CustomQuestion(
            title="How many House of Commons seats will the Conservative Party win in the 2025 Canadian general election?",
            description="Canadian Prime Minister Mark Carney, through Governor General Mary Simon, called new parliamentary elections for 28 April 2025 (BBC, CNN) [oai_citation_attribution:51‡gjopen.com](https://www.gjopen.com/questions/4174-how-many-house-of-commons-seats-will-the-conservative-party-win-in-the-2025-canadian-general-election/comments#:~:text=Canadian%20Prime%20Minister%20Mark%20Carney%2C,date%20will%20be%20adjusted%20accordingly).",
            resolution_criteria="Resolves to the number of seats won by the Conservative Party of Canada in the 2025 general election, as per the officially announced results [oai_citation_attribution:52‡gjopen.com](https://www.gjopen.com/questions/4174-how-many-house-of-commons-seats-will-the-conservative-party-win-in-the-2025-canadian-general-election/comments#:~:text=new%20parliamentary%20elections%20scheduled%20for,date%20will%20be%20adjusted%20accordingly).",
            fine_print="If the election is postponed, the question’s closing date will adjust accordingly [oai_citation_attribution:53‡gjopen.com](https://www.gjopen.com/questions/4174-how-many-house-of-commons-seats-will-the-conservative-party-win-in-the-2025-canadian-general-election/comments#:~:text=CNN%29,date%20will%20be%20adjusted%20accordingly). In case of ambiguity or legal challenges, the officially certified results will be used [oai_citation_attribution:54‡gjopen.com](https://www.gjopen.com/questions/4174-how-many-house-of-commons-seats-will-the-conservative-party-win-in-the-2025-canadian-general-election/comments#:~:text=new%20parliamentary%20elections%20scheduled%20for,date%20will%20be%20adjusted%20accordingly).",
            community_prediction=[0.0425, 0.3483, 0.4517, 0.1275, 0.0192, 0.0082, 0.0025],
            url="https://www.gjopen.com/questions/4174-how-many-house-of-commons-seats-will-the-conservative-party-win-in-the-2025-canadian-general-election",
            type="multiple_choice",
            options=["Fewer than 104", "Between 104 and 120", "Between 121 and 137", "Between 138 and 154", "Between 155 and 171", "Between 172 and 188", "189 or more"]
        ),
        CustomQuestion(
            title="Will Google’s search market share drop below 85% in 2025?",
            description="Google has long dominated the search engine market, holding around 90% share for years. This question asks whether new competition or technology will push Google’s monthly share below 85% at any point in 2025.",
            resolution_criteria="Resolves YES if Google’s search market share falls below 85% for any month in calendar year 2025, according to StatCounter data [oai_citation_attribution:56‡metaculus.com](https://www.metaculus.com/questions/31258/googles-search-market-share-below-85-in-2025/#:~:text=2025%3F%20www,according%20to%20Statcounter). If no month in 2025 drops below 85%, it resolves NO.",
            fine_print="Market share is measured by StatCounter’s global search engine market statistics. The threshold must be breached in at least one month of 2025.",
            community_prediction=0.10,
            url="https://www.metaculus.com/questions/31258/googles-search-market-share-below-85-in-2025/",
            type="binary"
        ),

        CustomQuestion(
            title="Which national team will win the UEFA Women’s Euro 2025 Final?",
            description="The UEFA Women’s Euro 2025 tournament will take place across Switzerland, beginning on 2 July 2025 with the final scheduled for 27 July 2025.",
            resolution_criteria="This will resolve to the national team that wins the Euro 2025 final (the tournament champion) on 27 July 2025.",
            fine_print="The outcome is determined by the official result of the UEFA Women’s Euro 2025 final.",
            community_prediction=[0.02, 0.19, 0.11, 0.18, 0.05, 0.05, 0.30, 0.07, 0.03],
            url="https://www.gjopen.com/questions/3900-which-national-team-will-win-the-uefa-women-s-euro-2025-final",
            type="multiple_choice",
            options=["Denmark", "England", "France", "Germany", "Italy", "Netherlands", "Spain", "Sweden", "Another team"]
        ),
        CustomQuestion(
            title="What will be the UK real GDP growth rate for the second quarter of 2025?",
            description="The UK economy is expected to see modest growth in 2025 despite persistent headwinds.",
            resolution_criteria="The question will be suspended on 30 June 2025 and resolved using the UK Office for National Statistics (ONS) data for Q2 2025 GDP growth, as reported in the 'GDP first quarterly estimate, UK: April to June 2025' (scheduled for release on 14 August 2025).",
            fine_print="The resolution will use the first official ONS release for Q2 2025. (Negative values indicate economic contraction.)",
            community_prediction=[0.04, 0.24, 0.62, 0.09, 0.01, 0.00],
            url="https://www.gjopen.com/questions/3869-what-will-be-the-uk-real-gdp-growth-rate-for-the-second-quarter-of-2025",
            type="multiple_choice",
            options=["Down >1.0%", "Down 0.0% to 1.0%", "Up 0.0% to 1.0%", "Up 1.0% to 2.0%", "Up 2.0% to 3.0%", "Up >3.0%"]
        ),
        CustomQuestion(
            title="How many seats will the Liberal-National Coalition win in the next Australian House of Representatives elections?",
            description="Australia must hold its next federal election for the House of Representatives by 27 September 2025. The incumbent Labor Party and the opposition Liberal-National Coalition will compete for control of the government.",
            resolution_criteria="This question will resolve after the next Australian general election (expected by late 2025). The outcome will be determined by how many seats the Liberal-National Coalition secures in the new 151-seat House: whether they obtain a majority (more than 50% of seats), a plurality (more seats than any other party but less than a majority), or neither.",
            fine_print="A 'majority' means at least 76 seats; a 'plurality' means the Coalition wins the most seats of any party but under 76 seats. If the Coalition merges or splits, or other extraordinary changes occur, the question will interpret outcomes in line with the intent (Coalition’s seat count).",
            community_prediction=[0.2236, 0.4102, 0.3662],
            url="https://www.gjopen.com/questions/3851-how-many-seats-will-the-liberal-national-coalition-the-coalition-win-in-the-next-australian-house-of-representatives-elections",
            type="multiple_choice",
            options=["Majority (76+ seats)", "Plurality (<76 seats but most)", "Neither"]
        ),
        CustomQuestion(
            title="In what place will Reform UK rank among political parties as of 5 September 2025, according to Politico?",
            description="Reform UK (led by Nigel Farage) has been polling strongly in the UK, challenging the established parties.",
            resolution_criteria="The result will be determined using Politico’s Poll of Polls for the UK as of 5 September 2025.",
            fine_print="We will use Politico’s 'Poll of Polls' chart data (using the 6-month range and Kalman filter) as of 5 Sept 2025. The percentage values will determine rank, regardless of graph ordering. If there’s a tie in support percentage, the party that most recently had lower support will be considered lower-ranked.",
            community_prediction=[0.2634, 0.4072, 0.2288, 0.1006],
            url="https://www.gjopen.com/questions/4161-in-what-place-will-reform-uk-rank-among-political-parties-as-of-5-september-2025-according-to-politico",
            type="multiple_choice",
            options=["1st", "2nd", "3rd", "4th place or lower"]
        ),
        CustomQuestion(
            title="What will be Starbucks’ revenues in China for Q2 2025 (Q3 of Starbucks’ fiscal year 2025)?",
            description="Starbucks has faced challenges in its China market and is implementing new strategies to improve performance.",
            resolution_criteria="The question will close on 30 June 2025 and be resolved using Starbucks’ official quarterly earnings report. Specifically, it will use the 'Revenues' figure for China in the FY2025 Q3 results.",
            fine_print="If Starbucks ceases to report China-specific revenues separately, or if an earnings release is delayed beyond a reasonable timeframe, the closest equivalent data will be used.",
            community_prediction=[0.22, 0.37, 0.30, 0.06, 0.02, 0.02, 0.01],
            url="https://www.gjopen.com/questions/3944-what-will-be-starbucks-revenues-in-china-for-the-second-quarter-of-2025-third-quarter-of-starbucks-fiscal-year-2025",
            type="multiple_choice",
            options=["< $660M", "$660M–$725M", "$725M–$790M", "$790M–$855M", "$855M–$920M", "≥ $920M", "Not reported"]
        ),
        CustomQuestion(
            title="Who will win the 2025 Polish presidential election?",
            description="Poland is scheduled to hold a presidential election in 2025, as incumbent President Andrzej Duda is term-limited. The election is expected in May 2025, with a runoff 14 days later if no candidate wins a majority in the first round.",
            resolution_criteria="This will resolve to the winner of the 2025 Polish presidential election. If a second-round runoff is required (likely in June 2025), the resolution will be based on the official runoff result (the candidate who becomes President-elect).",
            fine_print="An independent candidate endorsed by a party counts as that party’s candidate for this question. If a named candidate in an answer option is replaced by their party, the option will be updated to the new candidate.",
            community_prediction=[0.01, 0.05, 0.21, 0.72, 0.01],
            url="https://www.gjopen.com/questions/3942-who-will-win-the-2025-polish-presidential-election",
            type="multiple_choice",
            options=["Szymon Hołownia (Poland 2050)", "Sławomir Mentzen (Confederation)", "Karol Nawrocki (Law and Justice)", "Rafał Trzaskowski (Civic Coalition)", "Someone else"]
        ),
        CustomQuestion(
            title="What will be the average price of a dozen large eggs in the US in June 2025?",
            description="Egg prices in the United States have surged as avian influenza outbreaks led to the culling of millions of hens. Forecasters are asked to predict the average U.S. city retail price for a dozen Grade A large eggs in June 2025.",
            resolution_criteria="The question will be suspended on 30 June 2025 and resolved using official Bureau of Labor Statistics (BLS) price data for June 2025, as reported via the Federal Reserve Economic Data (FRED) database.",
            fine_print="The outcome will be determined by the BLS/FRED data release for June 2025. Price ranges in the answer options are defined in USD. If the BLS discontinues this specific series or significantly revises the methodology, the closest equivalent measure will be used.",
            community_prediction=[0.0766, 0.17, 0.2409, 0.1503, 0.1228, 0.1103, 0.0488, 0.0225, 0.0131, 0.0294, 0.0125, 0.0028],
            url="https://www.gjopen.com/questions/4106-what-will-be-the-average-price-of-a-dozen-large-eggs-in-the-us-in-june-2025",
            type="multiple_choice",
            options=["< $2.50", "$2.50–$2.99", "$3.00–$3.49", "$3.50–$3.99", "$4.00–$4.49", "$4.50–$4.99", "$5.00–$5.49", "$5.50–$5.99", "$6.00–$6.49", "$6.50–$6.99", "$7.00–$7.49", "$7.50 or more"]
        ),
        CustomQuestion(
                title="Will Maersk resume shipping in the Red Sea in 2025?",
                description="After a series of attacks in late 2023, Maersk rerouted vessels to avoid the Red Sea and isn’t in a rush to return amid ongoing security risks [oai_citation_attribution:58‡metaculus.com](https://www.metaculus.com/questions/31277/#:~:text=Metaculus%20www,routes%20around%20Africa%20for%20now).",
                resolution_criteria="Resolves YES if, before the end of 2025, Maersk (A.P. Moller-Maersk) announces or credible sources report that it has resumed sending ships through the Red Sea route [oai_citation_attribution:59‡manifold.markets](https://manifold.markets/WalterMartin/will-maersk-resume-shipping-in-the?play=true#:~:text=Will%20Maersk%20resume%20shipping%20in,report%20that%20shipping%20through). If no such resumption is reported by 2025, resolves NO.",
                fine_print="Any public announcement or credible report of Maersk resuming Red Sea transits counts. Absent that, the answer is NO.",
                community_prediction=0.60,
                url="https://www.metaculus.com/questions/31069/maersk-resume-shipping-in-red-sea-2025/",
                type="binary"
        ),
        CustomQuestion(
                title="Will there be at least 1,000 deaths due to direct conflict between Israel and Iran in 2025?",
                description="Although Israel and Iran engage in proxy conflicts, a direct war between them would mark a major escalation. This question asks if such a conflict in 2025 would result in 1,000 or more fatalities.",
                resolution_criteria="Resolves YES if credible sources report ≥1,000 deaths caused by direct military conflict between Israel and Iran during 2025. 'Direct conflict' means active hostilities directly between Israeli and Iranian forces (not via proxies).",
                fine_print="Casualty counts will rely on reputable media or official sources. If no direct Israel-Iran war occurs, or if fatalities remain below 1,000, it resolves NO.",
                community_prediction=0.08,
                url="https://www.metaculus.com/questions/30925/1000-deaths-due-to-israel-iran-conflict-in-2025/",
                type="binary"
        ),
        CustomQuestion(
                title="Will at least twice as many deportations by U.S. Immigration and Customs Enforcement occur in Fiscal Year 2025 compared to Fiscal Year 2024?",
                description="U.S. immigration enforcement has fluctuated in recent years. This question asks if ICE deportations (removals) in FY2025 will be at least double the number in FY2024.",
                resolution_criteria="Resolves YES if the number of deportations conducted by ICE in FY2025 is ≥2× the number in FY2024, based on official ICE/DHS statistics. Otherwise resolves NO.",
                fine_print="Fiscal Year 2025 runs Oct 1, 2024 – Sep 30, 2025. If FY2025 deportations are at least 200% of FY2024’s, that meets the criterion.",
                community_prediction=0.20,
                url="https://www.metaculus.com/questions/30954/ice-deportations-double-in-fy2025/",
                type="binary"
        ),
        CustomQuestion(
                title="Will mifepristone become significantly restricted or illegal to prescribe for abortions across the US before 2026?",
                description="Mifepristone, a drug used for medication abortions, has been the subject of legal battles over its FDA approval and availability.",
                resolution_criteria="Resolves YES if by Dec 31, 2025, a legal or regulatory action has taken effect that makes mifepristone largely unavailable or illegal to prescribe for abortion nationwide. If it remains generally available (even amid ongoing court cases), resolves NO.",
                fine_print="This refers to a nationwide ban or restriction (e.g. a final court ruling revoking approval). State-level restrictions or unresolved litigation do not count unless they result in de facto nationwide unavailability.",
                community_prediction=0.15,
                url="https://www.metaculus.com/questions/30950/nationwide-mifepristone-restriction-before-2026/",
                type="binary"
        ),
        CustomQuestion(
            title="Will a new war or a substantial escalation of an existing war kill at least 5,000 people in 2025?",
            description="There are several global flashpoints where conflict could erupt or intensify. This question asks whether any new war or major escalation in 2025 will result in 5,000 or more deaths during the year.",
            resolution_criteria="Resolves YES if at least one armed conflict that begins in 2025, or a major escalation of a pre-2025 conflict, causes ≥5,000 deaths in 2025 (per reputable sources). If not, resolves NO.",
            fine_print="Ongoing wars (like Ukraine) are counted only if a distinct, significant escalation is identified as starting in 2025. 'New war' means a conflict not already active at high intensity prior to 2025.",
            community_prediction=0.58,
            url="https://www.metaculus.com/questions/30929/new-war-or-escalation-5000-deaths-2025/",
            type="binary"
        ),
        CustomQuestion(
            title="Will US residents be able to order Starbucks delivery using a major AI interface before 2026?",
            description="Given current trends, there is a strong chance that at least one major AI interface will support ordering Starbucks delivery by the end of 2025 [oai_citation_attribution:65‡metaculus.com](https://www.metaculus.com/questions/31347/starbucks-delivery-possible-via-major-ai-interface-in-2025/#:~:text=Metaculus%20www,support%20Starbucks%20delivery%20before%202026).",
            resolution_criteria="Resolves YES if, by Dec 31, 2025, at least one widely-used AI assistant or interface (e.g. a popular chatbot or voice assistant) can directly facilitate ordering a Starbucks delivery in the US. Otherwise NO.",
            fine_print="The AI interface should be officially integrated with Starbucks or its delivery partners (e.g. an Alexa skill, Siri shortcut, or ChatGPT plugin enabling Starbucks orders). Beta features count as long as they are available to US users by end of 2025.",
            community_prediction=0.751,
            url="https://www.metaculus.com/questions/31347/starbucks-delivery-via-ai-interface-2025/",
            type="binary"
        ),
        CustomQuestion(
            title="Will semaglutide be taken off the FDA’s drug shortage list in 2025?",
            description="Semaglutide (sold under Ozempic, Wegovy, etc.) has been in high demand for diabetes and weight loss, leading to ongoing shortages.",
            resolution_criteria="Resolves YES if the FDA’s official Drug Shortages list no longer includes semaglutide at any point in 2025 (and it remains off the list thereafter). If semaglutide stays listed as in shortage through 2025, resolves NO.",
            fine_print="A removal that is reversed later in 2025 would still count (as it was taken off at some point). The FDA’s published shortage database will be the reference.",
            community_prediction=0.65,
            url="https://www.metaculus.com/questions/30961/semaglutide-off-shortage-list-2025/",
            type="binary"
        ),
        CustomQuestion(
            title="On December 31, 2025, will Google, Meta, Amazon, Tesla, or X accept cryptocurrency as a form of payment?",
            description="Several tech giants have dabbled in crypto payments (Tesla briefly accepted Bitcoin in 2021). This question asks if any of Google, Meta (Facebook), Amazon, Tesla, or X (Twitter) will be accepting cryptocurrency for payments by the end of 2025.",
            resolution_criteria="Resolves YES if at least one of the listed companies is officially accepting cryptocurrency as payment for its products or services as of Dec 31, 2025. If none of them do, resolves NO.",
            fine_print="Acceptance can be via direct or integrated third-party payment processors, as long as customers can pay in crypto. An announcement or policy in effect by end of 2025 is required.",
            community_prediction=0.51,
            url="https://www.metaculus.com/questions/30928/major-tech-company-crypto-payment-2025/",
            type="binary"
        ),
        CustomQuestion(
            title="Will an application to ban the Alternative für Deutschland (AfD) party be filed with Germany’s Federal Constitutional Court before 2026?",
            description="Germany’s domestic intelligence has classified AfD elements as extremist, prompting debate about possibly banning the far-right party. In early 2025, German parliamentarians debated a proposal to pursue a ban of the AfD [oai_citation_attribution:69‡politico.eu](https://www.politico.eu/article/alternative-for-germany-afd-ban-debate-far-right-german-election/#:~:text=German%20parliamentarians%20are%20set%20to,ahead%20of%20a%20national%20election), though many leaders (including Chancellor Scholz) voiced reservations [oai_citation_attribution:70‡politico.eu](https://www.politico.eu/article/alternative-for-germany-afd-ban-debate-far-right-german-election/#:~:text=But%20many%20mainstream%20politicians%20%E2%80%94,Scholz%20%E2%80%94%20have%20expressed%20reservations).",
            resolution_criteria="Resolves YES if an official application to ban the AfD is filed with Germany’s Federal Constitutional Court before January 1, 2026. If no such case is filed by then, resolves NO.",
            fine_print="This refers to initiating court proceedings to federally ban the AfD. Discussions or state-level actions don’t count unless they result in a formal federal court application.",
            community_prediction=0.20,
            url="https://www.metaculus.com/questions/30922/application-to-ban-afd-before-2026/",
            type="binary"
        ),
        CustomQuestion(
            title="Will Donald Trump attend Victory Day celebrations in Moscow, Russia, in 2025?",
            resolution_criteria="No additional criterion.",
            description="Russia has reported that it expects a \"special guest\" from the US for the 80th anniversary of Germany's defeat in World War II, known as Victory Day, on 9 May 2025 (Newsweek, The Moscow Times). Any physical presence in the city of Moscow on 9 May 2025 (local time) will be considered attendance.",
            fine_print="No additional fine print.",
            community_prediction=0.1707,
            url="https://www.gjopen.com/questions/4075-will-donald-trump-attend-victory-day-celebrations-in-moscow-russia-in-2025",
            type="binary"
        ),
        CustomQuestion(
            title="Before 1 October 2025, will negotiations on a China-Gulf Cooperation Council (GCC) free trade agreement be completed?",
            resolution_criteria="No additional criterion.",
            description="China and the GCC began negotiations for a free trade agreement in 2005, and the parties are reportedly pushing forward toward a conclusion despite concerns raised in May 2024 (Economist, Chinese State Council, GCC - News, US News & World Report, GCC - Member States). The signing of an FTA or the announcement of an agreement in principle (see EU-Mercosur announcement) would count, and ratification would be immaterial (e.g., US News & World Report, BBC).",
            fine_print="No additional fine print.",
            community_prediction=0.2783,
            url="https://www.gjopen.com/questions/4140-before-1-october-2025-will-negotiations-on-a-china-gulf-cooperation-council-gcc-free-trade-agreement-be-completed",
            type="binary"
        ),
        CustomQuestion(
            title="How many total Southwest Land Border Encounters will be reported in the US from June 2025 through August 2025 by US Customs and Border Protection (CBP)?",
            resolution_criteria="No additional criterion.",
            description="Cracking down on illegal immigration was a cornerstone of Donald Trump's 2024 election campaign, and his named \"border czar\" was already discussing plans for a major shift in US policy (Economist, NBC News, Arizona Mirror). The question will be suspended on 31 August 2025 and the outcome determined using data as reported by CBP for the calendar months of June 2025 through August 2025 once data for August 2025 are first reported (CBP). As of the launch of this question, CBP reported a total of 341,988 encounters in the US southwest between June 2024 and August 2024.",
            fine_print="No additional fine print.",
            community_prediction=[0.51, 0.24, 0.11, 0.06, 0.04, 0.02, 0.01, 0.01],
            url="https://www.gjopen.com/questions/4151-how-many-total-southwest-land-border-encounters-will-be-reported-in-the-us-from-june-2025-through-august-2025",
            type="multiple_choice",
            options=[
                "Fewer than 90,000",
                "At least 90,000, but fewer than 180,000",
                "At least 180,000, but fewer than 270,000",
                "At least 270,000, but fewer than 360,000",
                "At least 360,000, but fewer than 450,000",
                "At least 450,000, but fewer than 540,000",
                "At least 540,000, but fewer than 630,000",
                "630,000 or more"
            ]
        ),
        CustomQuestion(
            title="How many Registered Syrian Refugees will be in Turkey (aka Türkiye) as of 4 September 2025?",
            resolution_criteria="No additional criterion.",
            description="After the fall of the Assad regime in Syria, there is a question of when and at what pace the approximately three million refugees in Turkey may return home (Economist, AP, The Globalist). The question will be suspended on 3 September 2025 and the outcome determined using data as reported by the UN High Commissioner for Refugees (UNHCR) for \"Registered Syrian Refugees\" on its \"Syria Regional Refugee Response\" page for Turkey (UNHCR). As of 31 December 2024, a total of 2,901,478 registered Syrian refugees were in Turkey. If data for 4 September 2025 are not available, the total for the first date thereafter will be used for resolution.",
            fine_print="No additional fine print.",
            community_prediction=[
                0.0, 0.0, 0.0, 0.02, 0.05, 0.11, 0.22, 0.39, 0.17, 0.03, 0.01
            ],
            url="https://www.gjopen.com/questions/4149-how-many-registered-syrian-refugees-will-be-in-turkey-aka-turkiye-as-of-4-september-2025",
            type="multiple_choice",
            options=[
                "Fewer than 600,000",
                "At least 600,000, but fewer than 900,000",
                "At least 900,000, but fewer than 1.2 million",
                "At least 1.2 million, but fewer than 1.5 million",
                "At least 1.5 million, but fewer than 1.8 million",
                "At least 1.8 million, but fewer than 2.1 million",
                "At least 2.1 million, but fewer than 2.4 million",
                "At least 2.4 million, but fewer than 2.7 million",
                "At least 2.7 million, but fewer than 3.0 million",
                "At least 3.0 million, but fewer than 3.3 million",
                "3.3 million or more"
            ]
        ),
    CustomQuestion(
        title="Will a Chinese university be in the Top 10 of the QS World University Rankings for 2026?",
        resolution_criteria="This question resolves based on the official QS World University Rankings 2026, which will be published by QS Quacquarelli Symonds on June 19, 2025, at 00:01 BST. A “Yes” resolution occurs if any university located in mainland China is ranked within the Top 10 of the published list; otherwise, the question resolves to “No.”",
        description="The Quacquarelli Symonds (QS) World University Rankings compare over 1,500 universities around the world (QS Top Universities). In the rankings for 2025, the highest-ranked Chinese institution was Peking University at 14th, up from 17th for 2024 (Yahoo [Cision]). Rankings for 2026 are expected sometime in late spring or early summer 2025 (QS - World University Rankings).",
        fine_print="No additional fine print.",
        community_prediction=0.1744,
        url="https://www.gjopen.com/questions/4153-will-a-chinese-university-be-in-the-top-10-of-the-qs-world-university-rankings-for-2026",
        type="binary"
    ),

    CustomQuestion(
        title="What will be the value of Total Assets of the Federal Reserve as of 31 December 2025?",
        resolution_criteria="Resolves based on the Federal Reserve’s reported Total Assets on the balance sheet dated 31 Dec 2025 (from official Fed releases or FRED data) [oai_citation_attribution:23‡gjopen.com](https://www.gjopen.com/questions/3986-what-will-be-the-value-of-total-assets-of-the-federal-reserve-as-of-31-december-2025#:~:text=Group%20www,885963%20trillion) [oai_citation_attribution:24‡gjopen.com](https://www.gjopen.com/questions/3986-what-will-be-the-value-of-total-assets-of-the-federal-reserve-as-of-31-december-2025#:~:text=will%20be%20suspended%20on%2030,885963%20trillion). (The question will be suspended on 30 Dec 2025 and resolved when the data is available.)",
        description="The Federal Reserve’s balance sheet shrank by about $2 trillion from June 2022 to Dec 2024, though the pace of runoff slowed in mid-2024 [oai_citation_attribution:25‡gjopen.com](https://www.gjopen.com/questions/3986-what-will-be-the-value-of-total-assets-of-the-federal-reserve-as-of-31-december-2025#:~:text=The%20Federal%20Reserve%27s%20balance%20sheet,885963%20trillion). (As of 25 Dec 2024, Total Assets were approximately $6.886 trillion [oai_citation_attribution:26‡gjopen.com](https://www.gjopen.com/questions/3986-what-will-be-the-value-of-total-assets-of-the-federal-reserve-as-of-31-december-2025#:~:text=will%20be%20suspended%20on%2030,885963%20trillion).)",
        fine_print="No additional fine print.",
        community_prediction=[0.0160, 0.0110, 0.0410, 0.1780, 0.3440, 0.2050, 0.1700, 0.0350],
        url="https://www.gjopen.com/questions/3986",
        type="multiple_choice",
        options=[
            "Less than $5.4 trillion",
            "At least $5.4 trillion, but less than $5.7 trillion",
            "At least $5.7 trillion, but less than $6.0 trillion",
            "At least $6.0 trillion, but less than $6.3 trillion",
            "At least $6.3 trillion, but less than $6.6 trillion",
            "At least $6.6 trillion, but less than $6.9 trillion",
            "At least $6.9 trillion, but less than $7.2 trillion",
            "$7.2 trillion or more"
        ]
    ),

    CustomQuestion(
        title="Will California's 2018 Prop 12 (\"Prevention of Cruelty to Farm Animals Act\") be in effect on Jan 1, 2026?",
        resolution_criteria="Resolves YES if California’s Proposition 12 is still in legal effect on 1 January 2026. Resolves NO if it is struck down or not in force by that date [oai_citation_attribution:27‡metaculus.com](https://www.metaculus.com/c/future-perfect/31110/will-californias-2018-prop-12-be-in-effect-on-jan-1-2026/#:~:text=Metaculus%20www,effect%20on%20January%201%2C%202026).",
        description="California’s Proposition 12, a 2018 ballot initiative banning extreme confinement of farm animals, fully took effect on Jan 1, 2024 after surviving a Supreme Court challenge [oai_citation_attribution:28‡metaculus.com](https://www.metaculus.com/c/future-perfect/31110/will-californias-2018-prop-12-be-in-effect-on-jan-1-2026/#:~:text=Metaculus%20www,effect%20on%20January%201%2C%202026). This question asks if it will remain in effect through 2025.",
        fine_print="No additional fine print.",
        community_prediction=0.95,
        url="https://www.metaculus.com/questions/31110",
        type="binary"
    ),

    CustomQuestion(
        title="Will the bubble in the 'Magnificent Seven' stocks pop before 2026?",
        resolution_criteria="Resolves YES if at any point before Jan 1, 2026, at least four of the \"Magnificent Seven\" U.S. tech stocks (Apple, Microsoft, Alphabet, Amazon, Nvidia, Meta, Tesla) trade at closing prices at least 50% below their all-time highs [oai_citation_attribution:29‡manifold.markets](https://manifold.markets/mirrorbot/metaculus-will-the-bubble-in-the-ma?play=true#:~:text=This%20question%20will%20resolve%20positive,time%20high). (Share prices will be adjusted for any stock splits [oai_citation_attribution:30‡manifold.markets](https://manifold.markets/mirrorbot/metaculus-will-the-bubble-in-the-ma?play=true#:~:text=The%20question%20shall%20be%20resolved,source%20of%20share%20price%20data).) Otherwise resolves NO.",
        description="The top seven U.S. tech companies saw soaring valuations (the \"Magnificent Seven\"), leading to speculation of a bubble. This asks whether their stock prices will crash (halving from peak) by 2026.",
        fine_print="Data source: Yahoo Finance daily closing prices, adjusted for splits [oai_citation_attribution:31‡manifold.markets](https://manifold.markets/mirrorbot/metaculus-will-the-bubble-in-the-ma?play=true#:~:text=The%20question%20shall%20be%20resolved,source%20of%20share%20price%20data).",
        community_prediction=0.35,
        url="https://www.metaculus.com/questions/21531",
        type="binary"
    ),

    CustomQuestion(
        title="Before 2026, will OpenAI's commercial operations cease to be governed by its nonprofit board of directors?",
        resolution_criteria="Resolves YES if, by Jan 1, 2026, OpenAI’s for-profit arm is no longer under the control of the OpenAI nonprofit board (for example, if governance transfers to a new corporate board or other structure). Otherwise resolves NO.",
        description="OpenAI’s unusual governance (a nonprofit board overseeing a for-profit company) came under scrutiny after the tumultuous events of Nov 2023, when the board ousted and then reinstalled CEO Sam Altman [oai_citation_attribution:32‡metaforecast.org](https://metaforecast.org/questions/metaculus-20172#:~:text=The%20tumultuous%20events%20of%20November,and%20structure%20will%20remain%20intact). This question asks if OpenAI will fully shed nonprofit governance by 2026.",
        fine_print="No additional fine print.",
        community_prediction=0.63,
        url="https://www.metaculus.com/questions/20172",
        type="binary"
    ),

    CustomQuestion(
        title="Will an AI system achieve >85% on the FrontierMath benchmark before 2026?",
        resolution_criteria="Resolves YES if, by Jan 1, 2026, at least one AI system is documented to achieve greater than 85% performance on the FrontierMath advanced mathematics benchmark [oai_citation_attribution:33‡galois.com](https://www.galois.com/articles/o3-frontier-math-and-the-future-of-mathematics#:~:text=Frontier%20Math%20is%20designed%20to,on%20certain%20kinds%20of%20problems). Otherwise resolves NO.",
        description="FrontierMath is a challenging math competition benchmark introduced in 2023 to test AI reasoning. Early AI scores were around 2%, but OpenAI’s new model “o3” achieved about 25% in late 2024 [oai_citation_attribution:34‡galois.com](https://www.galois.com/articles/o3-frontier-math-and-the-future-of-mathematics#:~:text=Frontier%20Math%20is%20designed%20to,on%20certain%20kinds%20of%20problems) [oai_citation_attribution:35‡galois.com](https://www.galois.com/articles/o3-frontier-math-and-the-future-of-mathematics#:~:text=abilities%20in%20an%20AI,on%20certain%20kinds%20of%20problems). Will an AI reach >85% (near human-expert level) by 2026?",
        fine_print="No additional fine print.",
        community_prediction=0.15,
        url="https://www.metaculus.com/questions/35562",
        type="binary"
    ),

    CustomQuestion(
        title="Will an armed China-Taiwan conflict causing ≥100 deaths occur before 2026?",
        resolution_criteria="Resolves YES if by Jan 1, 2026 there is an armed conflict between the PRC (China) and ROC (Taiwan) that results in at least 100 fatalities [oai_citation_attribution:36‡metaculus.com](https://www.metaculus.com/questions/35272/invasion-of-taiwan-in-2025/#:~:text=Will%20armed%20conflicts%20between%20the,least%20100%20deaths%20before%202026). Otherwise resolves NO.",
        description="Tensions in the Taiwan Strait remain high, but no direct military conflict between China and Taiwan has happened in decades. This asks if a significant armed clash (100+ deaths) will occur by the end of 2025.",
        fine_print="No additional fine print.",
        community_prediction=0.06,
        url="https://www.metaculus.com/questions/11854",  # hypothetical URL placeholder
        type="binary"
    ),

    CustomQuestion(
        title="Will Iran possess a nuclear weapon before 2026?",
        resolution_criteria="This question will resolve YES if at least one of the following occurs before 2026: **(1)** Iran’s government announces it has a nuclear weapon and credible sources deem it credible; **(2)** Iran announces a nuclear weapon test and credible sources deem it credible; **(3)** an authoritative source (e.g. Arms Control Association, SIPRI, or FAS) reports that Iran has a nuclear weapon [oai_citation_attribution:37‡manifold.markets](https://manifold.markets/WalterMartin/will-iran-possess-a-nuclear-weapon#:~:text=1,the%20statement%20to%20be%20credible) [oai_citation_attribution:38‡manifold.markets](https://manifold.markets/WalterMartin/will-iran-possess-a-nuclear-weapon#:~:text=3,the%20Federation%20of%20American%20Scientists). A nuclear weapon is defined as a device for military use that causes a nuclear explosion; merely having weapons-grade material or related tech is not enough [oai_citation_attribution:39‡manifold.markets](https://manifold.markets/WalterMartin/will-iran-possess-a-nuclear-weapon#:~:text=A%20,not%20satisfy%20the%20resolution%20criteria). Resolves NO if none of the above by Jan 1, 2026.",
        description="Western intelligence has long suspected Iran of seeking nuclear weapons. This market asks whether Iran will actually obtain or test a nuclear weapon by the end of 2025.",
        fine_print="No additional fine print.",
        community_prediction=0.12,
        url="https://manifold.markets/WalterMartin/will-iran-possess-a-nuclear-weapon",
        type="binary"
    ),

    CustomQuestion(
        title="Will Israel strike Iran's nuclear facilities before the end of 2025?",
        resolution_criteria="Resolves YES if Israel conducts a military strike against Iran’s nuclear sites (e.g. enrichment facilities or reactors) before January 1, 2026. Otherwise resolves NO.",
        description="Israel has threatened to act militarily to stop Iran from getting nuclear weapons. This market forecasts whether Israel will carry out an attack on Iranian nuclear facilities by the end of 2025.",
        fine_print="No additional fine print.",
        community_prediction=0.50,
        url="https://manifold.markets/DanielFox/will-israel-strike-irans-nuclear-facilities-2025",
        type="binary"
    ),

    CustomQuestion(
        title="Will Tesla offer a Robo-Taxi (fully autonomous taxi) service by the end of 2025?",
        resolution_criteria="Resolves YES if, by Dec 31, 2025, Tesla launches a public ride-hailing service in which consumers can ride in fully self-driving Tesla vehicles (without a human driver or operator). Otherwise resolves NO.",
        description="Tesla has repeatedly promised a 'robo-taxi' service using its self-driving technology. This question asks if that promise will be realized by the end of 2025.",
        fine_print="No additional fine print.",
        community_prediction=0.40,
        url="https://manifold.markets/GabeGarboden/will-tesla-offer-a-robo-taxi-service-by-2025",
        type="binary"
    ),

    CustomQuestion(
        title="Will Bitcoin reach $200,000 before the end of 2025?",
        resolution_criteria="Resolves YES if the price of one Bitcoin is ≥$200,000 on any reputable exchange at any point before 11:59 PM (ET) on Dec 31, 2025 [oai_citation_attribution:41‡manifold.markets](https://manifold.markets/AviSchwartz/will-cryptocurrency-eg-bitcoin-reac?play=true#:~:text=Yes%3A%20Bitcoin%20reaches%20or%20exceeds,EST%20on%20December%2031%2C%202025). Resolves NO otherwise.",
        description="Bitcoin’s all-time high is around $69k (Nov 2021). Reaching $200k would roughly triple that record. This market forecasts whether Bitcoin will hit that milestone by end of 2025.",
        fine_print="No additional fine print.",
        community_prediction=0.08,
        url="https://manifold.markets/AviSchwartz/will-cryptocurrency-eg-bitcoin-reac",
        type="binary"
    ),

        CustomQuestion(
        title="Will Ukraine launch a nuclear weapons program before 2026?",
        resolution_criteria="Resolves YES if, before Jan 1, 2026, credible reporting confirms that Ukraine has initiated a program to develop or deploy nuclear weapons. (For the purposes of this question, a program includes deploying nuclear warheads obtained from an ally.) Otherwise resolves NO.",
        description="Russia’s invasion has prompted debates over whether Ukraine might pursue nuclear weapons, though Ukraine relies on security guarantees and gave up its Soviet-era nukes in 1994.",
        fine_print="No additional fine print.",
        community_prediction=0.01,
        url="https://www.metaculus.com/questions/17791",
        type="binary"
    ),


    CustomQuestion(
        title="Will we achieve Artificial General Intelligence (AGI) before 2026?",
        resolution_criteria="Resolves YES if by 31 Dec 2025 an AI system is widely recognized as having human-level general intelligence (for example, passing a high-quality Turing test or equivalent public milestone). Otherwise resolves NO.",
        description="Experts are divided on how soon AGI will arrive. This market gauges the chance that a true human-level AI is achieved (and acknowledged) before 2026.",
        fine_print="No additional fine print.",
        community_prediction=0.10,
        url="https://manifold.markets/RemNi/will-we-get-agi-before-2026",
        type="binary"
    ),

    CustomQuestion(
        title="Will Keir Starmer cease to be the prime minister of the United Kingdom before 1 October 2025?",
        resolution_criteria="No additional criterion.",
        description="Despite leading his Labour Party to a sweeping victory in the 2024 parliamentary elections, Starmer's popularity had cratered by the beginning of 2025 (Economist, MSN [GB News]).",
        fine_print="No additional fine print.",
        community_prediction=0.11,
        url="https://www.gjopen.com/questions/4079-will-keir-starmer-cease-to-be-the-prime-minister-of-the-united-kingdom-before-1-october-2025/crowd_forecast",
        type = "binary"
    ),

]
