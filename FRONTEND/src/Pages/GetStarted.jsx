import React, { useEffect, useState } from "react";
import { useAuth } from "../context/authContext";
import { Link, useNavigate } from "react-router-dom";
import Logo from "../assets/logo1.png";
import { FaBaseballBall } from "react-icons/fa";

const GameSection = () => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const [colorIndex, setColorIndex] = useState(0);
    const colors = ["#22c55e", "#3b82f6", "#eab308", "#ef4444", "#a855f7", "#f97316"];

    useEffect(() => {
        const interval = setInterval(() => {
            setColorIndex((prev) => (prev + 1) % colors.length);
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    const handleLogout = () => {
        logout();
        navigate("/");
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white">
            {/* Header Section */}
            <header className="flex items-center justify-between  px-2 lg:px-6 py-2 lg:py-4 bg-gray-800 shadow-lg">
                <Link to="/">
                    <img src={Logo} className="w-24 lg:w-24" alt="Logo" />
                </Link>
                <div className="flex items-center gap-4">
                    <img
                        className="rounded-full w-10 lg:w-12  object-cover border-2 border-white"
                        src={user?.photoURL || "https://via.placeholder.com/40"}
                        alt="Profile"
                    />
                    {user ? (
                        <button
                            onClick={handleLogout}
                            className="bg-red-500 hover:bg-red-600 px-5 py-2 rounded-md transition-all"
                        >
                            Logout
                        </button>
                    ) : (
                        <button
                            onClick={() => navigate("/login")}
                            className="bg-blue-500 hover:bg-blue-600 px-5 py-2 rounded-md transition-all"
                        >
                            Login
                        </button>
                    )}
                </div>
            </header>

            {/* Main Section */}
            <main className="flex flex-col items-center py-10 px-6 text-center">
                <h1 className="text-3xl lg:text-5xl font-bold mb-4">
                    Welcome back, {user?.displayName || "Guest"}!
                </h1>
                <h2 className="text-lg lg:text-2xl text-gray-300 mb-8">
                    Select your options to start playing your games
                </h2>

                {/* Options Section */}
                <div className="flex flex-col items-center bg-purple-800 p-6 rounded-lg shadow-lg w-full max-w-lg">
                    <div className="w-full flex flex-col gap-3">
                        <Link to="/score-card" className="w-full">
                            <button className="w-full bg-purple-500 hover:bg-purple-600 px-6 py-3 rounded-lg transition-all">
                                Get Score Card
                            </button>
                        </Link>
                        <Link to="/commentary-card" className="w-full">
                            <button className="w-full bg-purple-500 hover:bg-purple-600 px-6 py-3 rounded-lg transition-all">
                                Get Commentary Card
                            </button>
                        </Link>
                    </div>
                </div>

                {/* Animated Icon */}
                <div className="absolute bottom-10 left-8 text-6xl animate-bounce" style={{ color: colors[colorIndex] }}>
                    <FaBaseballBall />
                </div>
            </main>
        </div>
    );
};

export default GameSection;
