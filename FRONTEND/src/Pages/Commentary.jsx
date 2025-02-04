import React, { useEffect, useState } from "react";
import { Box, Button, FormControl, InputLabel, NativeSelect, TextField, Modal } from "@mui/material";
import { Link, useNavigate } from "react-router-dom";
import ArrowBack from "@mui/icons-material/ArrowBack";
import DefaultImage from "../assets/image.png";
import { FaBaseballBall } from "react-icons/fa";

const CommentaryCard = () => {
    const [responseData, setResponseData] = useState({
        key_moment: "N/A",
        sentiment_score: "N/A",
        summary: "N/A",
        audio: null,
        image: DefaultImage,
    });
    const [imageSrc, setImageSrc] = useState(DefaultImage);
    const [audioSrc, setAudioSrc] = useState(null);
    const [openModal, setOpenModal] = useState(false);
    const [playersList, setPlayersList] = useState([]);  // Players list from the API
    const [loadingPlayers, setLoadingPlayers] = useState(true);  // Loading state for players list
    const navigate = useNavigate();
    const [colorIndex, setColorIndex] = useState(0);
    const [team, setTeam] = useState("");
    const [player, setPlayer] = useState("");
    const [playerId, setPlayerId] = useState(null);
    const [language, setLanguage] = useState("");
    const [teamsList, setTeamsList] = useState([]);
    const [commentary, setCommentary] = useState("");
    const [error, setError] = useState("");

    const colors = ["#22c55e", "#3b82f6", "#eab308", "#ef4444", "#a855f7", "#f97316"];

    const fetchPlayers = async (teamId) => {
        if (!teamId) return;

        setLoadingPlayers(true);
        try {
            const response = await fetch(`http://35.206.83.91:8000/team_players/${teamId}`);
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

            const data = await response.json();
            setPlayersList(data.players || []);
        } catch (err) {
            setError("Failed to load players. Please try again.");
            console.error("Error fetching players:", err);
        } finally {
            setLoadingPlayers(false);
        }
    };

    useEffect(() => {
        const interval = setInterval(() => {
            setColorIndex((prev) => (prev + 1) % colors.length);
        }, 1000);
        return () => clearInterval(interval);
    }, [colors.length]);

    const handleCloseModal = () => {
        setOpenModal(false);
    };

    const handleTeamChange = (e) => {
        const teamId = parseInt(e.target.value, 10);
        setTeam(teamId);
        fetchPlayers(teamId);
    };

    const handlePlayerChange = (e) => {
        const selectedPlayerId = e.target.value;
        setPlayer(selectedPlayerId);
        setPlayerId(selectedPlayerId);
    };


    const handleStartGame = async () => {
        try {
            const response = await fetch('http://35.206.83.91:8000//process-commentary', { method: 'POST' }); // API request to fetch game data
            const data = await response.json();

            setResponseData({
                key_moment: data.key_moment || 'No key moment found',
                sentiment_score: data.sentiment_score || 'No sentiment score found',
                summary: data.summary || 'No summary available',
                imageSrc: data.imageSrc || '',  // Set image URL
                audioSrc: data.audioSrc || '',  // Set audio URL
            });

            // Redirect to the result page with the fetched data
            navigate('/game-result', { state: { data: responseData } });

        } catch (error) {
            console.error("Error fetching game data:", error);
            // Handle error appropriately here
        } finally {
            setLoading(false);
        }
    };


    return (
        <div className="min-h-screen bg-purple-100 flex justify-center items-center">
            <div className="w-full max-w-screen-lg p-5">
                {/* Outer Box with Shadow and Padding */}
                <Box
                    sx={{
                        width: "100%",
                        display: "flex",
                        bgcolor: "background.paper",
                        borderRadius: 2,
                        boxShadow: 3,
                        p: 3,
                    }}
                >
                    <div className="w-full">
                        <div className=" bg-green-500 shadow-lg rounded-lg w-10 p-2">
                            <Link to="/get-start">
                                <ArrowBack className="text-6xl text-white cursor-pointer  " />
                            </Link>
                        </div>
                        <main className="flex flex-col items-center py-10 px-3  relative font-bold">
                            <FaBaseballBall className="text-4xl lg:text-6xl animate-bounce" style={{ color: colors[colorIndex] }} />
                            <h4 className="text-2xl md:text-3xl lg:text-4xl text-black mb-10 text-center">
                                Select Options to Have Your Card for the Game
                            </h4>

                            {/* Card Section */}
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 w-full">
                                {/* Favorite Teams */}
                                <Box sx={{ width: "100%" }}>
                                    <FormControl fullWidth>
                                        <InputLabel variant="standard">Select favorite team</InputLabel>
                                        <NativeSelect value={team} onChange={handleTeamChange}>
                                            <option value="" disabled>Select favorite team</option>
                                            <option value="119">Los Angeles Dodgers</option>
                                            <option value="118">Kansas City Royals</option>
                                            <option value="120">Washington Nationals</option>
                                            <option value="121">New York Mets</option>
                                            <option value="117">Houston Astros</option>
                                            <option value="125">Indianapolis Blues</option>
                                        </NativeSelect>
                                    </FormControl>
                                </Box>

                                {/* Favorite Players */}
                                <Box sx={{ width: "100%" }}>
                                    <FormControl fullWidth>
                                        <InputLabel variant="standard">Select favorite player</InputLabel>
                                        <NativeSelect value={player} onChange={handlePlayerChange}>
                                            <option value="" disabled>Select favorite player</option>
                                            {loadingPlayers ? (
                                                <option>Loading players...</option>
                                            ) : playersList.length > 0 ? (
                                                playersList.map((p) => (
                                                    <option key={p.id} value={p.id}>{p.fullName}</option>
                                                ))
                                            ) : (
                                                <option>No players available</option>
                                            )}
                                        </NativeSelect>
                                    </FormControl>
                                </Box>

                                {/* Language Selection */}
                                <Box sx={{ width: "100%" }}>
                                    <FormControl fullWidth>
                                        <InputLabel variant="standard">Select a language</InputLabel>
                                        <NativeSelect value={language} onChange={(e) => setLanguage(e.target.value)}>
                                            <option value="" disabled>Select a language</option>
                                            <option value="en">English</option>
                                            <option value="es">Spanish</option>
                                            <option value="ja">Japanese</option>
                                            <option value="de">German</option>
                                        </NativeSelect>
                                    </FormControl>
                                </Box>
                            </div>

                            {/* Commentary Box (Spans Full Width) */}
                            <Box sx={{ width: "100%", mt: 6 }}>
                                <TextField
                                    label="Commentary"
                                    variant="outlined"
                                    multiline
                                    rows={5}
                                    value={commentary}
                                    onChange={(e) => setCommentary(e.target.value)}
                                    fullWidth
                                />
                            </Box>

                            {/* Start Game Button */}
                            <button
                                onClick={handleStartGame}
                                className="w-full sm:w-80 mt-6 focus:outline-none focus:ring-4 bg-blue-500 hover:bg-blue-300 text-white py-3 rounded-md text-lg font-medium transition duration-300"
                            >
                                Start your game
                            </button>
                        </main>
                    </div>
                </Box>

                {/* Modal */}
                <Modal open={openModal} onClose={handleCloseModal}>
                    <Box
                        sx={{
                            position: "absolute",
                            top: "50%",
                            left: "50%",
                            transform: "translate(-50%, -50%)",
                            width: 400,
                            bgcolor: "background.paper",
                            p: 4,
                            borderRadius: 2,
                            boxShadow: 3,
                        }}
                    >
                        <h2 className="text-xl font-semibold">User Information</h2>
                        <p><strong>Team:</strong> {team || "Not Selected"}</p>
                        <p><strong>Player:</strong> {player || "Not Selected"}</p>
                        <p><strong>Language:</strong> {language || "Not Selected"}</p>
                        <p><strong>Commentary:</strong> {commentary || ""}</p>
                        <Button onClick={() => setOpenModal(false)} variant="contained" className="mt-4">
                            Close
                        </Button>
                    </Box>
                </Modal>
            </div>
        </div>

    );
};

export default CommentaryCard;
