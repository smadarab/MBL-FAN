import { useState } from "react";
import { FaMicrophone, FaPaperclip, FaPaperPlane, FaBars, FaTimes } from "react-icons/fa";

const ChatSection = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);

    const toggleRecording = () => {
        setIsRecording((prevState) => !prevState);
    };

    const toggleSidebar = () => {
        setIsSidebarOpen((prevState) => !prevState);
    };

    return (
        <section className="h-screen flex bg-gray-900 text-white relative">
            {/* Sidebar */}
            <aside
                className={`fixed md:relative top-0 left-0 h-full bg-gray-800 p-4 flex flex-col transform transition-transform duration-300 z-10 ${isSidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
                    } w-3/4 md:w-1/4 lg:w-1/6`}
            >
                <button
                    className="md:hidden p-2 bg-gray-700 rounded-full hover:bg-gray-600 mb-4 self-end"
                    onClick={toggleSidebar}
                    title="Close Sidebar"
                >
                    <FaTimes size={18} />
                </button>
                <h2 className="text-lg font-semibold mb-6">Chat Friends</h2>
                <nav className="flex flex-col gap-4 flex-grow">
                    <button className="bg-gray-700 p-2 rounded hover:bg-gray-600">Option 1</button>
                    <button className="bg-gray-700 p-2 rounded hover:bg-gray-600">Option 2</button>
                    <button className="bg-gray-700 p-2 rounded hover:bg-gray-600">Option 3</button>
                </nav>
                <div className="mt-auto">
                    <button className="bg-gray-700 p-2 rounded hover:bg-gray-600 w-full">Footer Option</button>
                </div>
            </aside>

            {/* Sidebar Toggle Button at Bottom */}
            {!isSidebarOpen && (
                <div className="fixed bottom-20 left-10 transform -translate-x-1/2 z-20">
                    <button
                        className="p-4 bg-gray-700 rounded-full hover:bg-gray-600"
                        onClick={toggleSidebar}
                        title="Open Sidebar"
                    >
                        <FaBars size={24} />
                    </button>
                </div>
            )}

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col h-full md:ml-0">
                {/* Chat Messages */}
                <div className="flex-1 overflow-y-auto p-4">
                    <div className="mb-4">
                        <p className="bg-gray-700 p-3 rounded-tl-xl rounded-tr-xl rounded-br-xl inline-block max-w-[70%]">
                            Hello! How can I help you today?
                        </p>
                    </div>
                    <div className="mb-4 text-right">
                        <p className="bg-blue-600 p-3 rounded-tl-xl rounded-tr-xl rounded-bl-xl inline-block max-w-[70%]">
                            I need assistance with my project.
                        </p>
                    </div>
                </div>

                {/* Chat Input Section */}
                <div className="p-4 bg-gray-800 border-t border-gray-700">
                    <form className="flex items-center gap-2">
                        <button
                            type="button"
                            className="p-2 bg-gray-700 rounded-full hover:bg-gray-600 focus:outline-none"
                            title="Attach a file"
                        >
                            <FaPaperclip size={18} />
                        </button>

                        <button
                            type="button"
                            onClick={toggleRecording}
                            className={`p-2 rounded-full focus:outline-none ${isRecording ? "bg-red-600 text-white" : "bg-gray-700 hover:bg-gray-600"
                                }`}
                            title="Record voice"
                        >
                            <FaMicrophone size={18} />
                        </button>

                        <input
                            type="text"
                            placeholder="Type your message..."
                            className="flex-1 p-2 bg-gray-700 text-white rounded-md focus:outline-none focus:ring focus:ring-blue-500"
                        />

                        <button
                            type="submit"
                            className="p-2 bg-blue-600 hover:bg-blue-500 rounded-full focus:outline-none"
                            title="Send message"
                        >
                            <FaPaperPlane size={18} />
                        </button>
                    </form>
                </div>
            </div>
        </section>
    );
};

export default ChatSection;
