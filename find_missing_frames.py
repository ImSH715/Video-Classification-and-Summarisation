import os

# Path to your dataset directory (change this to your actual path)
dataset_path = "C:/Users/naya0/Uni/Dissertation/Video-Classification-and-Summarisation/data/UCF101"

# Read folder names (available action classes in your dataset)
available_classes = {name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))}

# Full list of 101 UCF101 action classes
ucf101_classes = {
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
    "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
    "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "Hammering",
    "HammerThrow", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
    "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
    "JugglingBalls", "JumpRope", "JumpingJack", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
    "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
    "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
    "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
    "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
    "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
    "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
    "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
    "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WallPushups", "WalkingWithDog", "WritingOnBoard",
    "YoYo"
}

# Compare and find missing classes
missing_classes = ucf101_classes - available_classes
extra_classes = available_classes - ucf101_classes

# Output results
print(f"Total available classes: {len(available_classes)}")
print(f"Missing classes ({len(missing_classes)}):")
for cls in sorted(missing_classes):
    print(f"{cls}")

if extra_classes:
    print(f"\n⚠️ Extra classes not part of UCF101 ({len(extra_classes)}):")
    for cls in sorted(extra_classes):
        print(f" - {cls}")
