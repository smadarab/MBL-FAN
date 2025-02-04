import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { Users, MessageCircle, Trophy, Handshake, Sparkles } from 'lucide-react';

const Community = () => {
    useEffect(() => {
        scrollTo(0, 0);
    }, []);

    const stats = [
        {
            icon: Users,
            label: "Active Members",
            value: "10,000+",
            description: "Passionate baseball fans from around the world"
        },
        {
            icon: MessageCircle,
            label: "Daily Interactions",
            value: "50,000+",
            description: "Engaging discussions and shared experiences"
        },
        {
            icon: Trophy,
            label: "Tournaments",
            value: "100+",
            description: "Monthly competitive events and prizes"
        },
        {
            icon: Handshake,
            label: "Partnerships",
            value: "25+",
            description: "Official MLB team collaborations"
        }
    ];





    return (
        <div className="min-h-screen ">
            {/* Hero Section */}
            <div className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8">
                <div className="absolute inset-0 overflow-hidden">
                    <div className="absolute -inset-[10px] opacity-50">
                        <div className="w-full h-full bg-gradient-to-r from-purple-500/30 via-blue-500/30 to-green-500/30 blur-3xl animate-pulse" />
                    </div>
                </div>

                <div

                    className="relative max-w-7xl mx-auto text-center"
                >
                    <h1
                        className="text-2xl lg:text-4xl font-bold text-transparent bg-clip-text  bg-gradient-to-r from-amber-500 to-red-500 mb-6"

                        data-aos="fade-right"
                        data-aos-delay="300"
                    >
                        Join Our Baseball Community
                    </h1>
                    <p
                        className="text-sm lg:text-lg text-gray-300 max-w-3xl mx-auto"
                        data-aos="fade-right"
                        data-aos-delay="300"
                    >
                        Connect with passionate baseball fans, share your insights, and participate in exclusive events.
                    </p>
                </div>
            </div>

            {/* Stats Grid */}
            <div
                className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20"
            >
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8"

                >
                    {stats.map((stat, index) => (
                        <div
                            key={index}

                            className="relative group"
                            data-aos="fade-right"
                            data-aos-delay="400"
                        >
                            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-xl blur-xl group-hover:blur-2xl transition-all duration-300" />
                            <div className="relative p-8 rounded-xl bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 hover:border-blue-500/50 transition-all duration-300">
                                <div className="flex items-center justify-center mb-4">
                                    <stat.icon className="w-8 h-8 text-blue-400" />
                                </div>
                                <h3 className="text-3xl font-bold text-white mb-2">{stat.value}</h3>
                                <p className="text-lg font-semibold text-gray-300 mb-2">{stat.label}</p>
                                <p className="text-gray-400">{stat.description}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Features Section */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20"

            >
                <div
                    className="text-center mb-16"

                >
                    <h2 className="text-2xl lg:text-4xl font-bold text-white mb-4"

                        data-aos="fade-right"
                        data-aos-delay="400"
                    >Community Features</h2>
                    <p className="text-gray-400"
                        data-aos="fade-right"
                        data-aos-delay="400"
                    >Everything you need to enhance your baseball experience</p>
                </div>

                <div
                    className="grid grid-cols-1 md:grid-cols-3 gap-8"

                >
                    {[
                        {
                            title: "Live Discussions",
                            description: "Real-time chat during games with fellow fans",
                            icon: MessageCircle
                        },
                        {
                            title: "Weekly Tournaments",
                            description: "Compete for prizes and recognition",
                            icon: Trophy
                        },
                        {
                            title: "Expert Analysis",
                            description: "Insights from baseball professionals",
                            icon: Sparkles
                        }
                    ].map((feature, index) => (
                        <div
                            key={index}

                            className="p-6 rounded-xl bg-gray-800/30 backdrop-blur-sm border border-gray-700/50 hover:border-purple-500/50 transition-all duration-300"
                            data-aos="fade-right"
                            data-aos-delay="500"
                        >
                            <feature.icon className="w-8 h-8 text-purple-400 mb-4" />
                            <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                            <p className="text-gray-400">{feature.description}</p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default Community;