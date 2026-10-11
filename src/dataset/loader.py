import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Curated reference samples from CNN / Daily Mail and general news articles
BENCHMARK_NEWS_SAMPLES = [
    {
        "id": "cnn_dm_01",
        "dataset": "CNN/DailyMail",
        "title": "NASA's James Webb Space Telescope reveals atmospheric features of distant exoplanet",
        "article": (
            "NASA's James Webb Space Telescope has captured a detailed molecular and chemical portrait of a distant world's skies. "
            "The telescope's array of highly sensitive instruments was trained on WASP-39 b, a 'hot Saturn' exoplanet orbiting a star "
            "roughly 700 light-years away from Earth. While Webb and other space telescopes, including Hubble and Spitzer, previously "
            "revealed isolated ingredients of this broiling planet's atmosphere, the new readings provide a full menu of atoms, "
            "molecules, and even signs of active chemistry and clouds. Among the groundbreaking findings is the first detection in an "
            "exoplanet atmosphere of sulfur dioxide, a molecule produced from chemical reactions triggered by high-energy light from the "
            "planet's parent star. On Earth, the protective ozone layer in the upper atmosphere is created in a similar way. "
            "The data also revealed sodium, potassium, and water vapor, confirming earlier observations while giving astronomers new clues "
            "into how these celestial bodies form and evolve over billions of years. Researchers described the precision of the data as a "
            "game changer for exoplanet science, opening doors to studying habitable rocky planets in the coming decades."
        ),
        "highlights": (
            "NASA's James Webb Space Telescope provides a complete molecular profile of exoplanet WASP-39 b. "
            "Researchers detect sulfur dioxide in an exoplanet atmosphere for the first time, signaling active photochemistry. "
            "Findings confirm water vapor and other elements, paving the way for habitable planet investigations."
        )
    },
    {
        "id": "cnn_dm_02",
        "dataset": "CNN/DailyMail",
        "title": "Global Climate Summit pledges new renewable energy targets by 2030",
        "article": (
            "Delegates from nearly 200 nations gathered in Dubai for the annual United Nations climate summit, concluding intense negotiations "
            "with a historic commitment to triple global renewable energy capacity by 2030. The agreement calls on countries to rapidly transition "
            "away from fossil fuels in energy systems in a just, orderly, and equitable manner, accelerating action in this critical decade to "
            "achieve net-zero carbon emissions by 2050. Proponents celebrated the inclusion of explicit transition language as a landmark victory, "
            "noting that previous treaties had hesitated to address coal, oil, and gas directly. However, several island nations and developing "
            "states voiced frustration, warning that the final text included loopholes that could delay urgent mitigation actions. "
            "Financing remained another major point of contention, with developing countries calling for trillions of dollars in grants rather "
            "than loans to help build solar and wind infrastructure. Despite the friction, multilateral financial institutions announced a $12 billion "
            "clean-tech fund to kick-start solar installations across Sub-Saharan Africa and Southeast Asia."
        ),
        "highlights": (
            "Nearly 200 nations pledge to triple renewable energy capacity by 2030 at the UN climate summit. "
            "The final pact explicitly calls for a transition away from fossil fuels toward net zero by 2050. "
            "Developing nations raise concerns regarding climate financing and implementation timelines."
        )
    },
    {
        "id": "news_article_03",
        "dataset": "News Articles",
        "title": "Breakthrough in solid-state batteries promises electric cars with 1000-km range",
        "article": (
            "Automotive engineers and materials scientists have unveiled a new prototype solid-state battery capable of propelling electric vehicles "
            "more than 1,000 kilometers on a single 15-minute charge. Traditional lithium-ion batteries rely on flammable liquid electrolytes, "
            "which limit energy density and pose thermal runaway hazards under extreme temperatures. The newly engineered cell replaces the liquid "
            "with an ultra-thin ceramic-polymer hybrid electrolyte paired with a pure lithium metal anode. Laboratory stress testing demonstrated "
            "over 1,200 continuous rapid charge-discharge cycles with less than five percent capacity degradation, far exceeding current commercial "
            "standards. Production experts caution that commercial scale-up remains a formidable hurdle due to the cleanroom manufacturing "
            "requirements for brittle ceramic separators. Nonetheless, two major automotive conglomerates announced joint venture pilot factories "
            "slated to begin low-volume vehicle integration trials early next year."
        ),
        "highlights": (
            "Scientists reveal a prototype solid-state battery offering electric vehicles a 1,000 km range per charge. "
            "The design uses a ceramic-polymer electrolyte and lithium anode, retaining 95% capacity after 1,200 cycles. "
            "Automakers plan pilot production runs next year despite ongoing manufacturing scalability challenges."
        )
    },
    {
        "id": "cnn_dm_04",
        "dataset": "CNN/DailyMail",
        "title": "Tech giants announce open standards consortium for artificial intelligence safety",
        "article": (
            "Leading artificial intelligence research labs and major technology corporations have banded together to establish the Frontier Model "
            "Safety Forum, an industry body aimed at establishing rigorous safety evaluations for next-generation frontier AI systems. "
            "The initiative comes amid mounting regulatory scrutiny in Europe and the United States over risks concerning cyber warfare, synthetic "
            "bio-threats, and autonomous deception. The consortium will establish standardized red-teaming methodologies, share vulnerability "
            "databases, and fund independent academic audits. While privacy advocates welcomed greater transparency, critics argued that self-regulation "
            "cannot replace legally binding government mandates. Legislative leaders in Washington affirmed that statutory compliance regimes and "
            "licensing requirements for models exceeding specified compute thresholds will remain on the congressional docket regardless of industry initiatives."
        ),
        "highlights": (
            "Major AI companies form the Frontier Model Safety Forum to set security standards for advanced systems. "
            "The group plans standardized red-teaming, vulnerability sharing, and independent academic research funding. "
            "Lawmakers emphasize that industry self-regulation will not substitute for upcoming federal statutory rules."
        )
    },
    {
        "id": "news_article_05",
        "dataset": "News Articles",
        "title": "Deep sea exploration identifies dozens of unknown marine species in the Clarion-Clipperton Zone",
        "article": (
            "A marine biology expedition navigating the abyssal plains of the central Pacific Ocean has uncovered dozens of previously undescribed species "
            "living at depths greater than 4,000 meters. Utilizing high-definition robotic submersibles, researchers surveyed the Clarion-Clipperton Zone, "
            "an underwater area rich in polymetallic nodules containing cobalt, nickel, and rare earth minerals targeted by mining consortia. "
            "Among the specimens documented were bioluminescent sea cucumbers, glass sponges with geometric silica skeletons, and predatory worms. "
            "Scientists emphasized that less than ten percent of abyssal biodiversity has been cataloged, warning that commercial seabed excavation "
            "could cause irreversible ecological damage before species are even formally classified. The International Seabed Authority is expected "
            "to review the scientific findings at its upcoming council meeting to deliberate environmental moratorium proposals."
        ),
        "highlights": (
            "Pacific deep-sea expedition discovers dozens of previously unknown species in the Clarion-Clipperton Zone. "
            "The region contains mineral-rich polymetallic nodules targeted for commercial seabed mining. "
            "Biologists warn mining could cause irreversible harm and urge conservation moratoria."
        )
    }
]

class DatasetLoader:
    """
    Loads benchmark news datasets:
    - Curated offline CNN/Daily Mail and News Article benchmark samples
    - HuggingFace datasets integration (`cnn_dailymail`, `3.0.0`) with streaming fallback
    """
    def __init__(self):
        pass

    @staticmethod
    def get_benchmark_samples() -> List[Dict[str, str]]:
        """Return the pre-cached CNN/Daily Mail and News Article benchmark set."""
        return BENCHMARK_NEWS_SAMPLES

    @staticmethod
    def load_from_hf_datasets(
        dataset_name: str = "cnn_dailymail",
        config_name: str = "3.0.0",
        split: str = "test",
        num_samples: int = 5
    ) -> List[Dict[str, str]]:
        """
        Load samples directly from HuggingFace datasets library.
        Uses streaming to avoid downloading multi-gigabyte datasets.
        Falls back to local benchmark if network or dataset fails.
        """
        try:
            from datasets import load_dataset
            logger.info(f"Streaming {num_samples} samples from Hugging Face '{dataset_name}' ({config_name}, {split})...")
            dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
            samples = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                samples.append({
                    "id": item.get("id", f"hf_{i}"),
                    "dataset": f"{dataset_name} ({config_name})",
                    "title": item.get("article", "")[:60] + "...",
                    "article": item.get("article", ""),
                    "highlights": item.get("highlights", "")
                })
            logger.info(f"Successfully loaded {len(samples)} samples from Hugging Face.")
            return samples
        except Exception as e:
            logger.warning(f"Could not load from Hugging Face ({e}). Falling back to local benchmark samples.")
            return BENCHMARK_NEWS_SAMPLES[:num_samples]
