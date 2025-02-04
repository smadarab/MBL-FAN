import { useEffect, useRef } from "react";
import Section from "../../components/Section";

import MoreSection from "./MoreSection";
import HeroIntroduction from "./HeroIntroduction";
import { Link } from "react-router-dom";
import FeaturesSection from "./FeaturesSection";


const HeroPage = () => {
    const parallaxRef = useRef(null);

    // useEffect(() => {
    //     scrollTo(0, 0)
    // }, [])

    return (
        <Section
            className="pt-[7rem] lg:pt-[8rem] -mt-[4rem]"
            crosses
            crossesOffset="lg:translate-y-[4rem]"

        >
            <div className="relative  lg:px-7" ref={parallaxRef} >
                <div className="relative z-1 max-w-[90%] mx-auto text-center mb-8 md:mb-16 lg:mb-[5rem]">
                    <div className="flex flex-col gap-5 lg:gap-14">
                        {/* Main Heading */}
                        <h1 className="text-xl md:text-2xl lg:text-5xl font-bold text-white uppercase  lg:mb-12"
                            data-aos="fade-up"
                            data-aos-delay="300"
                        >
                            Welcome to <span className="text-amber-500">Major </span>
                            <span className="text-red-500">Baseball</span>
                            <span className="text-blue-500"> League</span> Championship
                        </h1>

                        {/* First Section */}
                        <HeroIntroduction />

                        <hr className="border-gray-500 my-8" />


                        <section className=" bg-gradient-to-r from-gray-50 to-white text-gray-900 py-16 px-3"
                            data-aos="fade-left"
                            data-aos-delay="300"
                        >
                            <div className="max-w-5xl mx-auto text-center"
                                data-aos="fade-left"
                                data-aos-delay="500"
                            >
                                {/* Heading */}
                                <h2 className="text-xl md:text-3xl lg:text-4xl font-extrabold mb-6">
                                    Stay Connected with Us
                                </h2>

                                {/* Description */}
                                <p className="text-sm lg:text-lg font-medium mb-8 text-gray-900">
                                    Follow us on social media for the latest updates, exclusive content, and more. Let's build an engaged community together!
                                </p>

                                {/* Call-to-Action */}
                                <Link
                                    to="/community"
                                    className="inline-block bg-gradient-to-r from-blue-600 to-blue-500 text-white px-10 py-4 rounded-full font-semibold shadow-lg transform hover:-translate-y-1 hover:shadow-xl transition-all duration-300 text-base"
                                >
                                    Join the Community
                                </Link>
                            </div>
                        </section>

                        {/* Second Section */}
                        <hr className="border-gray-500  mt-10 lg-mt-20" />
                        <MoreSection />
                        <hr className="border-gray-500  mt-10 lg:mt-20" />
                        <FeaturesSection />
                        {/* <hr className="border-gray-500  mt-20" />
                        <UpcomingEventsSection /> */}
                    </div>
                </div>
            </div>
        </Section>
    );
};

export default HeroPage;
