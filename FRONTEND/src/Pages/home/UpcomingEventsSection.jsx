const UpcomingEventsSection = () => {
    const events = [
        {
            date: "January 20, 2025",
            time: "5:00 PM",
            teams: "Yankees vs. Red Sox",
        },
        {
            date: "January 22, 2025",
            time: "7:30 PM",
            teams: "Dodgers vs. Giants",
        },
        {
            date: "January 25, 2025",
            time: "4:00 PM",
            teams: "Mets vs. Cubs",
        },
    ];

    return (
        <section className="bg-gradient-to-r from-gray-50 to-white rounded-3xl text-gray-900 py-16 px-6">
            <div className="max-w-5xl mx-auto">
                <h2 className="text-3xl font-bold text-center mb-8">Upcoming Events</h2>
                <div className="timeline border-l-2 border-blue-500">
                    {events.map((event, index) => (
                        <div
                            key={index}
                            className="event relative pl-6 mb-8 flex flex-col md:flex-row md:items-center"
                        >
                            <div className="absolute -left-3 w-6 h-6 bg-blue-500 rounded-full border-2 border-white"></div>
                            <div className="text-sm text-gray-500 md:mr-6">
                                <p>{event.date}</p>
                                <p>{event.time}</p>
                            </div>
                            <div className="flex-1 bg-white p-4 rounded-lg shadow-md">
                                <h3 className="text-lg font-semibold">{event.teams}</h3>
                                <button className="bg-blue-600 text-white px-4 py-2 mt-2 rounded-lg font-medium hover:bg-blue-500 transition">
                                    View Details
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default UpcomingEventsSection;
