import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import HeroImageVedio from "../../assets/Hero-baseball-vedio.mp4";
import Button from "../../components/Button";
import { FaBaseballBall } from "react-icons/fa";

const StartSection = () => {
    const [colorIndex, setColorIndex] = useState(0);

    const colors = ["#22c55e", "#3b82f6", "#eab308", "#ef4444", "#a855f7", "#f97316"];

    useEffect(() => {
        const interval = setInterval(() => {
            setColorIndex((prev) => (prev + 1) % colors.length);
        }, 1000); // Change color every second

        return () => clearInterval(interval);
    }, [colors.length]);

    // Framer Motion variants
    const textVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.8, delay: 0.2 } },
    };

    const iconVariants = {
        hidden: { scale: 0 },
        visible: { scale: 1, transition: { duration: 0.6, ease: "easeInOut" } },
    };

    return (
        <motion.div
            initial="hidden"
            animate="visible"
            className="flex flex-col lg:flex-row items-center gap-8 mt-8 relative"

        >
            {/* Video Section */}
            <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: 0.3 }}
                className="flex-1 border p-6 lg:p-10 rounded-br-[60px] rounded-tl-[50px] shadow-lg bg-gray-900"


            >

                <div
                    data-aos="fade-in"
                    data-aos-delay="400"
                >
                    <video
                        src={HeroImageVedio}
                        autoPlay
                        loop
                        muted
                        className="rounded-lg shadow-lg w-full h-auto border border-gray-700"


                    />
                </div>
            </motion.div>

            {/* Text Section */}
            <motion.div variants={textVariants} className="flex-1 text-left">
                <h2 className="text-xl md:text-2xl lg:text-5xl  font-extrabold bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent"
                    data-aos="fade-left"
                    data-aos-delay="300"
                >
                    Major Baseball Gaming Project
                </h2>
                <p className="text-gray-300 text-[14px] lg:text-lg mt-6 leading-relaxed"
                    data-aos="fade-left"
                    data-aos-delay="400"
                >
                    Welcome to Google’s <span className="text-yellow-400 font-bold">Baseball Gaming</span> project, a platform built for fun, competition, and excitement. Dive into an immersive experience where you can:
                </p>
                <ul className="list-disc list-inside mt-4 text-gray-400 space-y-2 text-sm lg:text-lg"
                    data-aos="fade-left"
                    data-aos-delay="500"
                >
                    <li>Track your favorite teams and players.</li>
                    <li>Engage with live stats and game insights.</li>
                    <li>Challenge friends to thrilling gameplay.</li>
                </ul>
                <p className="text-gray-300 mt-4 text-sm lg:text-lg"
                    data-aos="fade-left"
                    data-aos-delay="600"
                >
                    Whether you're strategizing for a win or enjoying the thrill of the game, this platform has everything you need. Join the action today!
                </p>

                {/* Buttons */}
                {/* <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.5 }}
                    className="flex flex-col sm:flex-row gap-4 mt-8"
                >
                    <Button href="/features" className="bg-gradient-to-r from-green-900 to-blue-900 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all duration-300">
                        Explore Features
                    </Button>
                    <Button href="/register" className="bg-transparent border border-gray-500 text-gray-300 px-6 py-3 rounded-lg hover:border-green-400 hover:text-white transition-all duration-300">
                        Get Started
                    </Button>
                </motion.div> */}
            </motion.div>

            {/* Animated Icons */}
            <motion.div
                initial="hidden"
                animate="visible"
                variants={iconVariants}
                className="absolute -bottom-20 left-8 text-5xl hidden lg:block"
            >
                <FaBaseballBall style={{ color: colors[colorIndex] }} className="animate-bounce" />
            </motion.div>
            <motion.div
                initial="hidden"
                animate="visible"
                variants={iconVariants}
                className="absolute -top-12 right-8 text-5xl hidden lg:block"
            >
                <FaBaseballBall style={{ color: colors[(colorIndex + 2) % colors.length] }} className="animate-bounce" />
            </motion.div>
        </motion.div>
    );
};

export default StartSection;
