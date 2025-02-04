import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import DefaultImage from "../assets/image.png";
import Box from "@mui/material/Box";
import InputLabel from "@mui/material/InputLabel";
import FormControl from "@mui/material/FormControl";
import NativeSelect from "@mui/material/NativeSelect";
import Modal from "@mui/material/Modal";
import Button from "@mui/material/Button";
import { ArrowBack } from "@mui/icons-material";
import { FaBaseballBall } from "react-icons/fa";

const ScoreCard = () => {

    const [colorIndex, setColorIndex] = useState(0);
    const [team, setTeam] = useState("");
    const [player, setPlayer] = useState("");
    const [playerId, setPlayerId] = useState(null);
    const [language, setLanguage] = useState("");
    const [openModal, setOpenModal] = useState(false);
    const [teamsList, setTeamsList] = useState([]);
    const [playersList, setPlayersList] = useState([]);
    const [loadingPlayers, setLoadingPlayers] = useState(false);
    const [imageSrc, setImageSrc] = useState(DefaultImage);
    const [isLoadingImage, setIsLoadingImage] = useState(false);
    const [error, setError] = useState("");

    const colors = ["#22c55e", "#3b82f6", "#eab308", "#ef4444", "#a855f7", "#f97316"];

    // Fetch players when a team is selected
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

    const handleStartGame = async () => {
        if (!playerId) {
            alert("Please select a player before starting the game.");
            return;
        }

        setIsLoadingImage(true);
        setOpenModal(true);

        try {
            const response = await fetch(`http://35.206.83.91:8000/player_id/${playerId}`);
            if (!response.ok) throw new Error("Failed to fetch player image.");

            const data = await response.json();
            setImageSrc(data.image_url || DefaultImage);
        } catch (err) {
            console.error("Error fetching player image:", err);
            setImageSrc(DefaultImage);
        } finally {
            setIsLoadingImage(false);
        }
    };

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

    return (
        <div className="min-h-screen bg-purple-100 flex items-center justify-center p-5">
            <div className="w-full max-w-screen-lg">
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
                    <main className="relative font-bold w-full">
                        <div className="bg-green-500 shadow-lg rounded-lg w-10 p-2">
                            <Link to="/get-start">
                                <ArrowBack className="text-6xl text-white cursor-pointer" />
                            </Link>
                        </div>
                        <div className="flex flex-col items-center py-10 px-3">
                            <FaBaseballBall className="text-4xl lg:text-6xl animate-bounce" style={{ color: colors[colorIndex] }} />
                            <h4 className="text-2xl lg:text-4xl text-center !text-black mb-10">Select Options to have your card for the game</h4>

                            <div className="w-full">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <Box sx={{ m: 1, width: "100%" }}>
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

                                    <Box sx={{ m: 1, width: "100%" }}>
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

                                    <Box sx={{ m: 1, width: "100%" }}>
                                        <FormControl fullWidth>
                                            <InputLabel variant="standard">Select a language</InputLabel>
                                            <NativeSelect value={language} onChange={(e) => setLanguage(e.target.value)}>
                                                <option value="" disabled>Select a language</option>
                                                <option value="English">English</option>
                                            </NativeSelect>
                                        </FormControl>
                                    </Box>
                                </div>

                                <button onClick={handleStartGame} className="w-full mt-6 bg-blue-500 hover:bg-blue-600 text-white py-3 rounded-md text-lg font-medium transition duration-300">
                                    Start your game
                                </button>
                            </div>
                        </div>

                        <Modal open={openModal} onClose={handleCloseModal}>
                            <Box sx={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", width: 400, bgcolor: "background.paper", p: 4 }}>
                                <h2>User Information</h2>
                                <p><strong>Team:</strong> {team || "Not Selected"}</p>
                                <p><strong>Player:</strong> {player || "Not Selected"}</p>
                                <p><strong>Language:</strong> {language || "Not Selected"}</p>
                                {isLoadingImage ? (
                                    <div className="flex justify-center items-center">
                                        <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                                    </div>
                                ) : (
                                    <img src={imageSrc} alt="Player" className="w-full mt-4 rounded-lg" />
                                )}
                                <Button onClick={handleCloseModal} variant="contained">Close</Button>
                            </Box>
                        </Modal>
                    </main>
                </Box>
            </div>
        </div>
    );
};

export default ScoreCard;
