import { Link } from "react-router-dom";
import { FaFacebookF, FaTwitter, FaInstagram, FaYoutube } from "react-icons/fa";
import { MdContactMail, MdInfoOutline } from "react-icons/md";
import { GiBaseballBat } from "react-icons/gi";
import Logo from "../assets/logo1.png";

const Footer = () => {
    return (
        <div className="border-t border-gray-300 bg-gradient-to-b from-gray-900 to-gray-800 overflow-hidden text-gray-100">
            <div className="px-5 lg:px-7.5 py-10">
                {/* Top Section */}
                <div className="flex flex-col lg:flex-row lg:justify-between lg:items-center ">
                    <div className="flex flex-col items-center lg:items-start mb-5 lg:mb-0"
                        data-aos="fade-right"
                        data-aos-delay="300"
                    >
                        <Link to="/">
                            <img
                                src={Logo}
                                className="lg:w-40 w-24"
                                alt="Logo"
                            />
                        </Link>
                        <h2 className="mt-2 text-lg lg:text-2xl font-bold">
                            Major Baseball League
                        </h2>
                    </div>
                    {/* Social Media Links */}
                    <div className="flex space-x-4"
                        data-aos="fade-left"
                        data-aos-delay="400"
                    >
                        <Link to="#" className="hover:text-yellow-500">
                            <FaFacebookF size={20} />
                        </Link>
                        <Link to="#" className="hover:text-yellow-500">
                            <FaTwitter size={20} />
                        </Link>
                        <Link to="#" className="hover:text-yellow-500">
                            <FaInstagram size={20} />
                        </Link>
                        <Link to="#" className="hover:text-yellow-500">
                            <FaYoutube size={20} />
                        </Link>
                    </div>
                </div>

                {/* Middle Section */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mt-10"
                    data-aos="fade-left"
                    data-aos-delay="500"
                >
                    {/* About Section */}
                    <div>
                        <h3 className="text-xl font-semibold mb-3">About</h3>
                        <ul className="space-y-2">
                            <li>
                                <Link
                                    to="/about"
                                    className="hover:text-yellow-500 flex items-center"
                                >
                                    <MdInfoOutline className="mr-2" /> About Us
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/contact"
                                    className="hover:text-yellow-500 flex items-center"
                                >
                                    <MdContactMail className="mr-2" /> Contact
                                    Us
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/faq"
                                    className="hover:text-yellow-500 flex items-center"
                                >
                                    FAQ
                                </Link>
                            </li>
                        </ul>
                    </div>

                    {/* Features Section */}
                    <div>
                        <h3 className="text-xl font-semibold mb-3">Features</h3>
                        <ul className="space-y-2">
                            <li>
                                <Link
                                    to="/live-scores"
                                    className="hover:text-yellow-500 flex items-center"
                                >
                                    <GiBaseballBat className="mr-2" /> Live
                                    Scores
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/player-stats"
                                    className="hover:text-yellow-500 flex items-center"
                                >
                                    Player Statistics
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/insights"
                                    className="hover:text-yellow-500 flex items-center"
                                >
                                    Game Insights
                                </Link>
                            </li>
                        </ul>
                    </div>

                    {/* Resources Section */}
                    <div>
                        <h3 className="text-xl font-semibold mb-3">
                            Resources
                        </h3>
                        <ul className="space-y-2">
                            <li>
                                <Link
                                    to="/schedule"
                                    className="hover:text-yellow-500"
                                >
                                    Game Schedule
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/blog"
                                    className="hover:text-yellow-500"
                                >
                                    Blog/Articles
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/highlights"
                                    className="hover:text-yellow-500"
                                >
                                    Video Highlights
                                </Link>
                            </li>
                        </ul>
                    </div>

                    {/* Policies Section */}
                    <div>
                        <h3 className="text-xl font-semibold mb-3">Policies</h3>
                        <ul className="space-y-2">
                            <li>
                                <Link
                                    to="/privacy"
                                    className="hover:text-yellow-500"
                                >
                                    Privacy Policy
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/terms"
                                    className="hover:text-yellow-500"
                                >
                                    Terms of Service
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/cookies"
                                    className="hover:text-yellow-500"
                                >
                                    Cookie Policy
                                </Link>
                            </li>
                        </ul>
                    </div>
                </div>

                {/* Bottom Section */}
                <div className="mt-10 text-center border-t border-gray-700 pt-5"
                    data-aos="fade-in"
                    data-aos-delay="600"
                >
                    <p className="text-gray-200">
                        © {new Date().getFullYear()} Major Baseball League. All
                        rights reserved.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Footer;
