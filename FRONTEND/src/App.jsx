import { Outlet } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import CookieBanner from "./components/CookieBanner"; // Import CookieBanner
import { useState, useEffect } from "react";
import AOS from "aos";
import "aos/dist/aos.css"
const App = () => {
    const [smokeStyle, setSmokeStyle] = useState({});

    const handleMouseMove = (e) => {
        const { clientX, clientY } = e;
        setSmokeStyle({
            left: `${clientX}px`,
            top: `${clientY}px`,
        });
    };

    useEffect(() => {
        // Add event listener for mouse movement
        window.addEventListener("mousemove", handleMouseMove);
        return () => {
            // Clean up event listener when the component unmounts
            window.removeEventListener("mousemove", handleMouseMove);
        };
    }, []);


    useEffect(() => {
        AOS.init({
            offset: 100,
            duration: 800,
            easing: "ease-in-sine",
            delay: 100,
        });
        AOS.refresh();
    }, []);

    return (
        <div className="relative pt-[4.75rem] lg:pt-[5.25rem] overflow-auto">
            {/* Smoke Effect */}
            <div
                className="fixed w-12 h-12 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-full pointer-events-none blur-lg opacity-80 transform -translate-x-1/2 -translate-y-1/2 transition-transform duration-150"
                style={{
                    position: "fixed",
                    transform: "translate(-50%, -50%)",
                    ...smokeStyle,
                }}
            ></div>

            {/* Cookie Banner */}
            <CookieBanner /> {/* Ensures it appears on all pages */}

            <Header />
            <Outlet />
            <Footer />
        </div>
    );
};

export default App;
