import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import DefaultImage from "../assets/image.png"; // Import the default image

const GameResult = () => {
    const location = useLocation();
    const { data } = location.state || {};  // Destructure the passed data

    if (!data) {
        return <div className='min-h-screen bg-slate-50'>
            <div className='flex items-center justify-center flex-col gap-5 h-[100vh] '>
                <h1 className='text-black font-bold lg:text-5xl text-2xl'> No game data found! </h1>
                <Link to="/" className='text-black font-semibold border p-3 rounded-lg'>Back to the home page</Link>
            </div>
        </div>;
    }

    return (
        <div className="min-h-screen bg-slate-50 p-4 md:p-6 lg:p-8">
            <div className="max-w-4xl mx-auto">
                <div className="bg-white rounded-lg shadow-md overflow-hidden">
                    {/* Header */}
                    <div className="border-b border-slate-200 p-6">
                        <h1 className="text-2xl md:text-3xl font-bold text-slate-900">
                            Game Results
                        </h1>
                    </div>

                    {/* Content */}
                    <div className="p-6 space-y-6">
                        {/* Key Moment Section */}
                        <div className="space-y-2">
                            <h2 className="text-lg font-semibold text-slate-900">Key Moment</h2>
                            <p className="text-slate-600">
                                {data.key_moment}
                            </p>
                        </div>

                        {/* Sentiment Score Section */}
                        <div className="space-y-2">
                            <h2 className="text-lg font-semibold text-slate-900">Sentiment Score</h2>
                            <p className="text-slate-600">
                                {data.sentiment_score}
                            </p>
                        </div>

                        {/* Summary Section */}
                        <div className="space-y-2">
                            <h2 className="text-lg font-semibold text-slate-900">Summary</h2>
                            <p className="text-slate-600">
                                {data.summary}
                            </p>
                        </div>

                        {/* Image Section */}
                        <div className="space-y-2">
                            <h2 className="text-lg font-semibold text-slate-900">Image</h2>
                            <div className="overflow-hidden rounded-lg bg-slate-100">
                                <img
                                    src={data.imageSrc || DefaultImage}
                                    alt="Game"
                                    className="w-full h-auto object-cover"
                                />
                            </div>
                        </div>

                        {/* Audio Section */}
                        <div className="space-y-2">
                            <h2 className="text-lg font-semibold text-slate-900">Audio</h2>
                            <div className="bg-slate-100 rounded-lg p-4">
                                <p className="text-slate-600 text-center">
                                    {data.audioSrc ? (
                                        <audio controls className="w-full max-w-[500px] mx-auto">
                                            <source src={data.audioSrc} type="audio/wav" />
                                            Your browser does not support the audio element.
                                        </audio>
                                    ) : (
                                        <p>No audio available.</p>
                                    )}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default GameResult;
