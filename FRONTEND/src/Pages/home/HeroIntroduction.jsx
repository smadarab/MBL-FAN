

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { CircleDot, Circle } from 'lucide-react';

import HeroImageDes from "../../assets/Hero-baseball-Image.png";
import { Link } from 'react-router-dom';



const HeroIntroduction = () => {
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    const [firstColorBall, setFirstColorBall] = useState(0);

    const colors = ['#22c55e', '#3b82f6', '#eab308', '#ef4444', '#a855f7', '#f97316'];

    useEffect(() => {
        const interval = setInterval(() => {
            setFirstColorBall(prev => (prev + 1) % colors.length);
        }, 2000);

        const handleMouseMove = (e) => {
            setMousePosition({ x: e.clientX, y: e.clientY });
        };

        window.addEventListener('mousemove', handleMouseMove);
        return () => {
            clearInterval(interval);
            window.removeEventListener('mousemove', handleMouseMove);
        };
    }, []);

    return (
        <div className="relative min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 overflow-hidden px-3 lg:px-8">
            {/* Animated background elements */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -inset-[10px] opacity-50">
                    <div className="w-full h-full bg-gradient-to-r from-blue-500/30 via-purple-500/30 to-pink-500/30 blur-3xl animate-pulse" />
                </div>
            </div>

            {/* Main content */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className="relative pt-10 lg:pt-32 flex flex-col lg:flex-row  gap-12"
            >
                {/* Text content */}
                <div className="flex-1 space-y-8">
                    <motion.h2
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.3 }}
                        className="text-xl md:text-3xl lg:text-4xl font-bold bg-gradient-to-r text-left  from-green-400 to-blue-500 bg-clip-text text-transparent"
                    >
                        Enjoy Gaming with Friends All Over the Country
                    </motion.h2>

                    <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.6 }}
                        className="text-sm lg:text-lg text-gray-300 leading-relaxed text-left"
                    >
                        Get real-time MLB updates, player stats, and game insights all in one place! Stay connected with your favorite teams and players while competing and having fun with friends.
                    </motion.p>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.9 }}
                        className="flex gap-4 flex-col lg:flex-row "
                    >
                        <Link to="/register" className="px-8 py-3 bg-gradient-to-r from-green-400 to-blue-500 rounded-full text-white font-semibold hover:shadow-lg hover:shadow-green-500/30 transition-all duration-300">
                            Get Started
                        </Link>
                        <Link to="/features" className="px-8 py-3 border border-gray-600 rounded-full text-white font-semibold hover:border-green-400 transition-all duration-300">
                            Learn More
                        </Link>
                    </motion.div>

                </div>

                {/* Image section */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.6 }}
                    className="flex-1 relative"
                    style={{
                        transform: `perspective(1000px) rotateY(${(mousePosition.x - window.innerWidth / 2) * 0.02}deg) rotateX(${(mousePosition.y - window.innerHeight / 2) * -0.02}deg)`
                    }}
                >
                    <div className="relative rounded-2xl overflow-hidden bg-gradient-to-r from-gray-800 to-gray-700 p-1">
                        <div className="absolute inset-0 bg-gradient-to-r from-green-400/20 to-blue-500/20 blur-xl" />
                        <img
                            src={HeroImageDes}
                            alt="Baseball Gaming"
                            className="relative rounded-xl w-full h-auto shadow-2xl"
                        />
                    </div>
                </motion.div>

                {/* Animated ball icons */}
                <motion.div
                    animate={{
                        rotate: 360
                    }}
                    transition={{
                        duration: 20,
                        repeat: Infinity,
                        ease: "linear"
                    }}
                    className="absolute bottom-10 left-10 opacity-30"
                >
                    <CircleDot size={40} color={colors[firstColorBall]} />
                </motion.div>

                <motion.div
                    animate={{
                        rotate: -360
                    }}
                    transition={{
                        duration: 15,
                        repeat: Infinity,
                        ease: "linear"
                    }}
                    className="absolute top-20 right-20 opacity-30"
                >
                    <Circle size={30} color={colors[(firstColorBall + 2) % colors.length]} />
                </motion.div>
            </motion.div>

            {/* Stats section */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2 }}
                className="absolute bottom-0 left-0 right-0 bg-gray-800/50 backdrop-blur-sm hidden lg:block"
            >
                <div className="max-w-7xl mx-auto py-8 px-4 grid grid-cols-2 lg:grid-cols-4 gap-8">
                    {[
                        { label: "Active Players", value: "1000+" },
                        { label: "Daily Matches", value: "500+" },
                        { label: "Live Games", value: "50+" },
                        { label: "Happy Users", value: "10K+" }
                    ].map((stat, index) => (
                        <div key={index} className="text-center">
                            <div className="text-2xl font-bold text-green-400">{stat.value}</div>
                            <div className="text-gray-200">{stat.label}</div>
                        </div>
                    ))}
                </div>
            </motion.div>
        </div>
    );
};

export default HeroIntroduction;