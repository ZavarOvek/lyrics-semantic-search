"""SPEC-04 §3: synthetic demo corpus generator.

Generates ~200 fully fabricated (non-copyrighted) songs across 20 themes,
10 title/artist variants each, built from a per-theme line bank (verse/
chorus/bridge lines sampled without replacement, one seeded RNG for
reproducibility). None of this text is derived from or resembles any real
song -- every line was composed from scratch for this project.

Runs the standard build pipeline (ingest -> preprocess -> embed -> index)
against the generated songs via an in-memory Source, writing everything
under data_demo/demo/. Uses embedder=tfidf, index=numpy -- CPU-only, no
model download, no GPU, no network call -- same choice
tests/test_end_to_end_search.py makes against the golden fixture, and for
the same reason: a demo should build in seconds on any machine.

Usage (idempotent -- reruns are cache-skipped by run_build's own
freshness check unless --force is passed):
    python scripts/generate_demo_corpus.py [--force] [--data-root data_demo]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

sys.path.insert(0, ".")

from lyrics_search.config import ExperimentConfig, RetrievalConfig
from lyrics_search.contracts import RawSong
from lyrics_search.core.text import make_song_id
from lyrics_search.runners.build import run_build

SEED = 20260730  # arbitrary, fixed -- reproducible generation across runs

# Each theme: (title_base, verse_lines, chorus_lines, bridge_lines).
# All text below is original, fabricated for this project -- see module
# docstring and notes/reports/spec04-report.md's §3 section.
THEMES: list[tuple[str, list[str], list[str], list[str]]] = [
    ("Sunrise Valley", [
        "Sunrise Valley waking up so slow",
        "Golden light is spilling through the trees",
        "Morning birds are singing on the wind",
        "Dew is shining on the fields of green",
        "I can hear the valley come alive",
        "Every hill is glowing in the dawn",
    ], [
        "Sunrise Valley calling out my name",
        "Wake up wake up to another day",
        "Sunrise Valley never fades away",
        "Light is breaking over every hill",
        "Morning morning fills my heart with hope",
        "Sunrise Valley shining on and on",
    ], [
        "Somewhere past the ridge the sun keeps rising",
        "Nothing in this valley stays the same",
        "I will follow every road at daybreak",
        "Sunrise Valley holds my heart forever",
    ]),
    ("Midnight Highway", [
        "Headlights cutting through the midnight air",
        "Radio is humming something low",
        "Miles of empty asphalt stretch ahead",
        "City lights are fading in the mirror",
        "Windows down and engine running steady",
        "Midnight highway carries me away",
    ], [
        "Midnight highway take me where I'm going",
        "Driving driving under starless skies",
        "Midnight highway never lets me sleep",
        "Keep the wheels turning till the morning light",
        "Midnight highway carries all my dreams",
        "Driving on into the great unknown",
    ], [
        "Somewhere down this road I lose my worries",
        "Every mile erases yesterday",
        "Midnight highway keeps on calling louder",
        "I will drive until the engine stops",
    ]),
    ("Ocean Drift", [
        "Waves are rolling gently to the shore",
        "Salty wind is tangled in my hair",
        "Seagulls circle overhead at noon",
        "Tide is pulling secrets from the sand",
        "Sunlight dances on the rolling water",
        "Ocean drift is carrying me home",
    ], [
        "Ocean ocean drifting far away",
        "Let the water carry all my fears",
        "Ocean drift beneath the endless sky",
        "Waves keep crashing softly on the shore",
        "Ocean ocean pulling me back in",
        "Drifting drifting where the current leads",
    ], [
        "Deep beneath the surface secrets hide",
        "Every wave erases every footprint",
        "Ocean drift will never let me go",
        "I belong forever to the tide",
    ]),
    ("Autumn Ember", [
        "Leaves are turning orange red and gold",
        "Autumn ember burning in the trees",
        "Cool wind rustles through the fallen leaves",
        "Smoke is rising from the chimney fire",
        "Pumpkins line the porches down the street",
        "Autumn ember fading into dusk",
    ], [
        "Autumn ember glowing in the twilight",
        "Falling falling leaves upon the ground",
        "Autumn ember warms this quiet evening",
        "Season changing colors all around",
        "Autumn ember holding on to summer",
        "Slowly slowly fading into gray",
    ], [
        "Every leaf remembers where it started",
        "Autumn ember never burns too bright",
        "Soon the frost will settle on the meadow",
        "Autumn ember keeps the cold at bay",
    ]),
    ("Winter Hollow", [
        "Snow is falling soft on winter hollow",
        "Frozen breath is hanging in the air",
        "Icicles are hanging from the rooftops",
        "Footprints vanish underneath the snowfall",
        "Fire crackles warm inside the cabin",
        "Winter hollow silent white and still",
    ], [
        "Winter hollow wrapped in fallen snow",
        "Cold and quiet blankets every hill",
        "Winter hollow holds the world asleep",
        "Snowflakes drifting slowly to the ground",
        "Winter hollow waiting for the spring",
        "Frozen frozen underneath the moonlight",
    ], [
        "Somewhere underneath the ice it's waiting",
        "Winter hollow keeps its secrets buried",
        "Spring will come and melt away the silence",
        "Winter hollow sleeps beneath the frost",
    ]),
    ("Neon Skyline", [
        "City lights are blinking in the distance",
        "Neon skyline painted red and blue",
        "Traffic hums a rhythm down below",
        "Rooftops glow beneath the passing clouds",
        "Skyscrapers reaching up to touch the night",
        "Neon skyline never goes to sleep",
    ], [
        "Neon skyline shining through the smoke",
        "Lights are flashing on the crowded street",
        "Neon skyline pulsing with the night",
        "City never rests beneath the stars",
        "Neon skyline calling out my name",
        "Bright lights burning till the break of dawn",
    ], [
        "Somewhere in the crowd I lose my shadow",
        "Neon skyline swallows every sound",
        "Every window holds another story",
        "Neon skyline keeps the city moving",
    ]),
    ("Wildflower Field", [
        "Wildflowers scatter across the open field",
        "Bees are humming softly in the breeze",
        "Tall grass bending underneath the sunlight",
        "Butterflies are dancing over petals",
        "Meadow stretching far beyond the fences",
        "Wildflower field beneath a summer sky",
    ], [
        "Wildflower field where all my worries fade",
        "Growing growing wild beneath the sun",
        "Wildflower field forever in my memory",
        "Colors blooming softly in the wind",
        "Wildflower field calling me back home",
        "Petals falling gently to the ground",
    ], [
        "Every flower knows the way to sunlight",
        "Wildflower field remembers every season",
        "Roots run deep beneath the quiet meadow",
        "Wildflower field will bloom again next spring",
    ]),
    ("Silver Rain", [
        "Silver rain is falling on the rooftop",
        "Thunder rumbles far beyond the hills",
        "Puddles gather on the empty sidewalk",
        "Lightning flashes bright across the sky",
        "Streetlights flicker underneath the storm",
        "Silver rain is washing off the dust",
    ], [
        "Silver rain keeps falling through the night",
        "Every drop is tapping on the window",
        "Silver rain will wash away the sorrow",
        "Storm clouds gather heavy overhead",
        "Silver rain is dancing on the pavement",
        "Falling falling softly on the town",
    ], [
        "Somewhere in the storm I find my shelter",
        "Silver rain will pass before the morning",
        "Every cloud eventually clears away",
        "Silver rain leaves everything renewed",
    ]),
    ("Desert Mirage", [
        "Heat is rising off the endless sand",
        "Desert mirage shimmers in the distance",
        "Cactus shadows stretch across the dunes",
        "Wind is carving patterns in the dust",
        "Sun beats down upon the empty highway",
        "Desert mirage fading with the dusk",
    ], [
        "Desert mirage dancing in the heat",
        "Nothing but the sand and endless sky",
        "Desert mirage calling me to wander",
        "Miles of silence stretching out ahead",
        "Desert mirage holds a hidden oasis",
        "Chasing chasing shadows in the sun",
    ], [
        "Somewhere past the dunes the water's waiting",
        "Desert mirage plays its ancient tricks",
        "Every step reveals another horizon",
        "Desert mirage never stays the same",
    ]),
    ("Mountain Echo", [
        "Mountain echo bouncing off the peaks",
        "Climbing higher through the morning mist",
        "Rocky paths are winding up the ridge",
        "Eagles circle far above the valley",
        "Snow caps glisten in the fading light",
        "Mountain echo calling from the summit",
    ], [
        "Mountain echo carries through the canyon",
        "Climbing climbing closer to the sky",
        "Mountain echo never fades away",
        "Every step brings me closer to the top",
        "Mountain echo shouting to the valley",
        "Reaching reaching for the highest peak",
    ], [
        "Somewhere near the summit clouds are gathering",
        "Mountain echo answers every call",
        "Every peak reveals another mountain",
        "Mountain echo guides me to the top",
    ]),
    ("Firelight Circle", [
        "Firelight flickers in the gathering dark",
        "Friends are singing songs beneath the stars",
        "Sparks are rising up into the night",
        "Stories passed around the crackling flames",
        "Marshmallows roasting slowly on the fire",
        "Firelight circle warm against the cold",
    ], [
        "Firelight circle glowing through the night",
        "Gather round and let the stories flow",
        "Firelight circle keeps us close together",
        "Sparks are dancing high into the sky",
        "Firelight circle burning till the morning",
        "Voices rising with the crackling flame",
    ], [
        "Somewhere in the smoke the memories linger",
        "Firelight circle never lets us drift",
        "Every spark remembers where it started",
        "Firelight circle keeps the cold away",
    ]),
    ("Paper Airplane", [
        "Paper airplane sailing through the classroom",
        "Folded dreams are drifting on the breeze",
        "Childhood windows open to the sunshine",
        "Paper airplane spiraling toward the ground",
        "Laughter echoes down the empty hallway",
        "Paper airplane carries yesterday",
    ], [
        "Paper airplane flying far away",
        "Carried carried on a summer wind",
        "Paper airplane holding all my wishes",
        "Folded gently made to touch the sky",
        "Paper airplane never really lands",
        "Flying flying back to yesterday",
    ], [
        "Somewhere in the sky my wishes travel",
        "Paper airplane knows no boundaries",
        "Every fold remembers where we started",
        "Paper airplane carries me back home",
    ]),
    ("Electric Pulse", [
        "Bass is thumping through the crowded room",
        "Lights are flashing to the driving rhythm",
        "Bodies moving to the electric pulse",
        "Speakers shaking every wall around",
        "Dancers spinning underneath the strobe",
        "Electric pulse is running through the crowd",
    ], [
        "Electric pulse is beating through the night",
        "Feel the rhythm moving through your bones",
        "Electric pulse will never let you stop",
        "Dancing dancing till the break of dawn",
        "Electric pulse is pulling everyone",
        "Lights are burning bright into the morning",
    ], [
        "Somewhere in the bass the crowd is rising",
        "Electric pulse keeps building to the peak",
        "Every beat is pulling harder closer",
        "Electric pulse will never fade away",
    ]),
    ("Quiet Harbor", [
        "Boats are resting gently in the harbor",
        "Morning fog is rolling off the water",
        "Fishermen are mending nets at sunrise",
        "Waves are lapping softly on the dock",
        "Lighthouse standing steady through the night",
        "Quiet harbor sleeping in the mist",
    ], [
        "Quiet harbor holds me safe from storms",
        "Resting resting where the water's calm",
        "Quiet harbor never lets me drown",
        "Every boat comes home before the dark",
        "Quiet harbor keeps the world at bay",
        "Peaceful peaceful underneath the stars",
    ], [
        "Somewhere past the breakwater it's calmer",
        "Quiet harbor holds the tide at rest",
        "Every wave grows softer near the shoreline",
        "Quiet harbor welcomes me back home",
    ]),
    ("Golden Hour", [
        "Sun is dipping low behind the hills",
        "Golden hour paints the sky in amber",
        "Shadows stretching long across the meadow",
        "Warm light settling on the quiet town",
        "Birds are heading home before the dusk",
        "Golden hour fading into evening",
    ], [
        "Golden hour holds the day a little longer",
        "Light is fading slow across the sky",
        "Golden hour keeps the warmth a little closer",
        "Everything looks softer in this light",
        "Golden hour never lasts quite long enough",
        "Fading fading into shades of blue",
    ], [
        "Somewhere on the horizon colors linger",
        "Golden hour slips away too soon",
        "Every evening brings another sunset",
        "Golden hour promises to return",
    ]),
    ("Stargazer", [
        "Stars are scattered far across the darkness",
        "Stargazer counting every point of light",
        "Milky way is stretching overhead",
        "Telescope is pointed at the heavens",
        "Night is quiet underneath the cosmos",
        "Stargazer lost among the stars",
    ], [
        "Stargazer reaching for the galaxy",
        "Watching watching lights so far away",
        "Stargazer never feels so small",
        "Universe is spinning overhead",
        "Stargazer chasing every falling star",
        "Dreaming dreaming underneath the night",
    ], [
        "Somewhere out beyond the furthest starlight",
        "Stargazer wonders what is waiting",
        "Every constellation tells a story",
        "Stargazer keeps on searching for the truth",
    ]),
    ("Broken Compass", [
        "Compass needle spinning round in circles",
        "Lost among the paths that lead nowhere",
        "Maps are folded crumpled in my pocket",
        "Every road looks different in the fog",
        "Searching for a signpost in the darkness",
        "Broken compass leading me astray",
    ], [
        "Broken compass can't decide direction",
        "Wandering wandering without a plan",
        "Broken compass leaves me second guessing",
        "Every path could lead me home or further",
        "Broken compass spins but never settles",
        "Lost lost somewhere on the way",
    ], [
        "Somewhere past the fog a light is waiting",
        "Broken compass still points somewhere true",
        "Every wrong turn teaches me the right one",
        "Broken compass finally finds the north",
    ]),
    ("Velvet Curtain", [
        "Velvet curtain rising on the stage",
        "Spotlight finds me standing in the silence",
        "Audience is waiting in the darkness",
        "Footlights flicker on the wooden floor",
        "Orchestra is tuning in the pit",
        "Velvet curtain trembles in the draft",
    ], [
        "Velvet curtain opens on the story",
        "Stepping stepping into golden light",
        "Velvet curtain never stays closed forever",
        "Every scene deserves its moment bright",
        "Velvet curtain falling at the ending",
        "Bowing bowing to the crowd tonight",
    ], [
        "Somewhere past the curtain lies another story",
        "Velvet curtain hides a thousand dreams",
        "Every actor waits behind the fabric",
        "Velvet curtain rises one more time",
    ]),
    ("River Bend", [
        "River winding slowly through the valley",
        "Current pulling gently toward the bend",
        "Willow branches trailing in the water",
        "Fish are swimming underneath the ripples",
        "Stones are smooth beneath the flowing water",
        "River bend beneath the afternoon",
    ], [
        "River bend keeps flowing to the ocean",
        "Carried carried by the gentle stream",
        "River bend will never stop its journey",
        "Water finds a way around the stone",
        "River bend beneath the willow shadows",
        "Flowing flowing on to somewhere new",
    ], [
        "Somewhere past the bend the water widens",
        "River bend remembers every rainfall",
        "Every drop eventually finds the ocean",
        "River bend will carry me along",
    ]),
    ("Glass House", [
        "Glass house standing fragile in the sunlight",
        "Every window shows another secret",
        "Walls are thin but somehow always holding",
        "Light is bending softly through the panes",
        "Careful footsteps echo on the floorboards",
        "Glass house trembles when the wind blows hard",
    ], [
        "Glass house built on nothing but a promise",
        "Fragile fragile but it's standing still",
        "Glass house holds together through the storms",
        "Every crack still lets the sunlight in",
        "Glass house shining brighter than it's breaking",
        "Standing standing though it's built on glass",
    ], [
        "Somewhere in the cracks the light gets brighter",
        "Glass house never really falls apart",
        "Every storm just makes the walls grow stronger",
        "Glass house holds the both of us together",
    ]),
]

ARTIST_ADJECTIVES = [
    "Silver", "Crimson", "Electric", "Golden", "Velvet", "Hollow", "Wild",
    "Quiet", "Neon", "Amber", "Frozen", "Restless", "Distant", "Faded",
    "Radiant", "Broken", "Hidden", "Endless", "Painted", "Drifting",
]
ARTIST_NOUNS = [
    "Wolves", "Doves", "Tide", "Echoes", "Skyline", "Horizon", "Embers",
    "Wanderers", "Signals", "Static", "Hollow", "Circuit", "Compass",
    "Harbor", "Ravens", "Meridian", "Lanterns", "Current", "Orbit", "Field",
]
TITLE_SUFFIXES = [
    "", " II", " Reprise", " Anthem", " at Night", " Forever", " Again",
    " Rising", " in the Rain", " Remix",
]


def _make_artist(rng, used: set[str]) -> str:
    while True:
        name = f"{rng.choice(ARTIST_ADJECTIVES)} {rng.choice(ARTIST_NOUNS)}"
        if name not in used:
            used.add(name)
            return name


def _make_song_text(rng, verse_lines, chorus_lines, bridge_lines, include_bridge: bool) -> str:
    verse = rng.sample(verse_lines, k=min(3, len(verse_lines)))
    chorus = rng.sample(chorus_lines, k=min(3, len(chorus_lines)))
    parts = ["[Verse]\n" + "\n".join(verse), "[Chorus]\n" + "\n".join(chorus)]
    if include_bridge and bridge_lines:
        bridge = rng.sample(bridge_lines, k=min(2, len(bridge_lines)))
        parts.append("[Bridge]\n" + "\n".join(bridge))
    return "\n\n".join(parts)


def generate_songs() -> list[RawSong]:
    import random

    rng = random.Random(SEED)
    used_artists: set[str] = set()
    used_titles: set[str] = set()
    songs: list[RawSong] = []

    for title_base, verse_lines, chorus_lines, bridge_lines in THEMES:
        for i, suffix in enumerate(TITLE_SUFFIXES):
            title = f"{title_base}{suffix}"
            assert title not in used_titles, f"duplicate title: {title!r}"
            used_titles.add(title)
            artist = _make_artist(rng, used_artists)
            text = _make_song_text(
                rng, verse_lines, chorus_lines, bridge_lines, include_bridge=(i % 2 == 0)
            )
            songs.append(RawSong(
                song_id=make_song_id(artist, title),
                artist=artist,
                title=title,
                text_raw=text,
                source="synthetic:spec04-demo-corpus",
                is_translation=False,
            ))
    return songs


@dataclass
class _InMemorySource:
    songs: list[RawSong]

    def iter_songs(self) -> Iterator[RawSong]:
        yield from self.songs


def main() -> None:
    parser = argparse.ArgumentParser(description="SPEC-04 §3 demo corpus generator")
    parser.add_argument("--data-root", default="data_demo")
    parser.add_argument("--force", action="store_true", help="bypass every stage's cache")
    args = parser.parse_args()

    songs = generate_songs()
    print(f"Generated {len(songs)} synthetic songs across {len(THEMES)} themes.")

    config = ExperimentConfig(
        corpus="demo", embedder="tfidf", index="numpy",
        retrieval=RetrievalConfig(mode="hybrid", top_k=50, return_n=10),
    )
    run_build(config, data_root=args.data_root, force=args.force, source=_InMemorySource(songs))


if __name__ == "__main__":
    main()
