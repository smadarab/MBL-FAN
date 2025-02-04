import { FaChartLine, FaUserFriends, FaBaseballBall } from "react-icons/fa";
import { Link } from "react-router-dom";

const FeaturesSection = () => {
    const features = [
        {
            icon: <FaChartLine className="text-blue-500 text-4xl mb-4" />,
            title: "Real-time MLB Updates",
            description:
                "Stay updated with live game stats and never miss a moment of the action.",
        },
        {
            icon: <FaBaseballBall className="text-green-500 text-4xl mb-4" />,
            title: "Player Stats",
            description:
                "Track your favorite players and teams with in-depth statistics and insights.",
        },
        {
            icon: <FaUserFriends className="text-yellow-500 text-4xl mb-4" />,
            title: "Gaming with Friends",
            description:
                "Compete, chat, and have fun with friends in thrilling gameplay experiences.",
        },
    ];

    return (
        <section className="bg-gradient-to-r from-gray-50 to-white rounded-3xl text-gray-900 py-16 px-6"
            data-aos="fade-in"
            data-aos-delay="300"
        >
            <div className="max-w-6xl mx-auto"
            >
                <h2 className="text-xl md:text-2xl lg:text-4xl font-bold text-center mb-8"
                    data-aos="fade-right"
                    data-aos-delay="400"
                >Features about user</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8"
                    data-aos="fade-right"
                    data-aos-delay="400"
                >
                    {features.map((feature, index) => (
                        <div
                            key={index}
                            className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300 text-center border"
                        >
                            {feature.icon}
                            <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                            <p className="text-gray-600">{feature.description}</p>
                        </div>
                    ))}
                </div>

                <Link
                    to="/features"
                    className="mt-10 inline-block bg-gradient-to-r from-blue-600 to-blue-500 text-white px-10 py-4 rounded-full font-semibold shadow-lg transform hover:-translate-y-1 hover:shadow-xl transition-all duration-300"
                    data-aos="fade-right"
                    data-aos-delay="400"
                >
                    More Features
                </Link>
            </div>
        </section>
    );
};

export default FeaturesSection;
