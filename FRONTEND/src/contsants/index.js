import { MdTipsAndUpdates } from "react-icons/md";
import { FaBaseballBall } from "react-icons/fa";
import { GiGamepadCross } from "react-icons/gi";
import { MdOutlinePersonalInjury } from "react-icons/md";
import { BsChatLeftText } from "react-icons/bs";
import { GiLevelFourAdvanced } from "react-icons/gi";
import { FaUsers, FaCommentDots, FaTrophy, FaHandshake } from "react-icons/fa";

export const navigation = [
  {
    id: "0",
    title: "Home",
    url: "/",
  },
  {
    id: "1",
    title: "Features",
    url: "/features",
  },
  {
    id: "3",
    title: "Community",
    url: "/community",
  },
];

export const features = [
  {
    id: 1,
    icon: MdTipsAndUpdates,
    color: "text-blue-500",
    title: "Real-Time Updates",
    description:
      "Stay ahead of the game with live updates on scores, game highlights, and player performances.",
  },
  {
    id: 2,
    icon: FaBaseballBall,
    color: "text-red-500",
    title: "Player Stats",
    description:
      "Dive deep into player statistics, compare performances, and track your favorite players' progress.",
  },
  {
    id: 3,
    icon: GiGamepadCross,
    color: "text-green-500",
    title: "Interactive Game Insights",
    description:
      "Analyze game strategies with detailed breakdowns and insights powered by advanced analytics.",
  },
  {
    id: 4,
    icon: MdOutlinePersonalInjury,
    color: "text-yellow-500",
    title: "Personalized Experience",
    description:
      "Customize your dashboard to follow specific teams, players, and game schedules.",
  },
  {
    id: 5,
    icon: BsChatLeftText,
    color: "text-purple-500",
    title: "Chat with Fans",
    description:
      "Connect with other baseball enthusiasts to share game predictions and excitement.",
  },
  {
    id: 6,
    icon: GiLevelFourAdvanced,
    color: "text-pink-500",
    title: "Advanced Tools",
    description: null,
    sublist: [
      "AI-powered game predictions",
      "Fantasy League Tracker",
      "Virtual Coaching with MLB experts",
    ],
  },
];

export const communityInfo = [
  {
    id: 1,
    icons: FaUsers,
    color: "text-blue-500",
    title: "Connect with Fans",
    subText: ` Meet and interact with baseball enthusiasts from all over the world.`,
  },
  {
    id: 2,
    icons: FaCommentDots,
    color: "text-green-500",
    title: "Discussions",
    subText: `Share your thoughts, predictions, and game strategies in lively forums.`,
  },
  {
    id: 3,
    icons: FaTrophy,
    color: "text-yellow-500",
    title: "Competitions",
    subText: `Participate in fantasy leagues and community challenges to win rewards.`,
  },
  {
    id: 4,
    icons: FaHandshake,
    color: "text-purple-500",
    title: "Collaborations",
    subText: `Team up with others for game analysis, discussions, and more.`,
  },
];
