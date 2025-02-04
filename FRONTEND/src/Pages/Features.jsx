import React, { useEffect } from 'react';

import Section from "../components/Section";
import Button from "../components/Button";
import { features } from '../contsants';

const Features = () => {
    useEffect(() => {
        scrollTo(0, 0);
    }, []);


    return (
        <Section
            className="pt-[8rem] -mt-[4rem]"
            crosses
            crossesOffset="lg:translate-y-[4rem]"
        >
            <div className="relative">
                {/* Animated background gradient */}
                <div className="absolute inset-0 overflow-hidden -z-10">
                    <div className="absolute -inset-[10px] opacity-30">
                        <div className="w-full blur-3xl animate-pulse" />
                    </div>
                </div>

                {/* Feature Header */}
                <div

                    className="flex flex-col items-center gap-8 px-6 lg:px-20 max-w-7xl mx-auto"
                >
                    <h2 className="text-xl md:text-3xl lg:text-4xl font-bold text-center"
                        data-aos="fade-right"
                        data-aos-delay="300"
                    >
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-amber-500 to-red-500">
                            Discover Our Features
                        </span>
                    </h2>
                    <p

                        className="text-sm lg:text-lg text-gray-300 text-center max-w-3xl"

                        data-aos="fade-right"
                        data-aos-delay="300"
                    >
                        Experience baseball like never before with advanced tools, exclusive insights, and a vibrant community of fans.
                    </p>

                    {/* Features Grid */}
                    <div

                        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 w-full mt-12"
                    >
                        {features.map((item) => (
                            <div
                                key={item.id}

                                className="group relative"

                                data-aos="fade-up"
                                data-aos-delay="400"
                            >
                                {/* Background Glow */}
                                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl blur-xl group-hover:blur-2xl transition-all duration-300" />

                                {/* Feature Card */}
                                <div className="relative p-8 rounded-xl bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 hover:border-blue-500/50 transition-all duration-300">
                                    {/* Icon */}
                                    {item.icon && <item.icon className={`w-12 h-12 ${item.color} mb-6`} />}

                                    {/* Title */}
                                    <h3 className="text-xl font-bold text-white mb-4">{item.title}</h3>

                                    {/* Description */}
                                    {item.description && (
                                        <p className="text-gray-300 mb-4">{item.description}</p>
                                    )}

                                    {/* Sublist */}
                                    {item.sublist && (
                                        <ul className="list-disc list-inside text-gray-300 space-y-2">
                                            {item.sublist.map((sub, index) => (
                                                <li key={index} className="pl-4">{sub}</li>
                                            ))}
                                        </ul>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>

                </div>

                {/* Call-to-Action Section */}
                <div

                    className="mt-20 text-center px-6"

                    data-aos="fade-right"
                    data-aos-delay="400"
                >
                    <div className="max-w-2xl mx-auto bg-gradient-to-r from-gray-800/50 to-gray-900/50 p-12 rounded-2xl backdrop-blur-sm border border-gray-700/50">
                        <h3 className="text-xl md:text-3xl lg:text-4xl font-bold text-white mb-6 ">
                            Ready to Experience Baseball Like Never Before?
                        </h3>
                        <Button
                            href="/register"
                            className="flex items-center  px-8 py-4 bg-gradient-to-r from-amber-500 to-red-500 text-white rounded-full font-semibold hover:shadow-lg hover:shadow-amber-500/25 transition-all duration-300 "
                        >
                            <span> Get Started Now</span>

                        </Button>
                    </div>
                </div>
            </div>
        </Section>
    );
};

export default Features;