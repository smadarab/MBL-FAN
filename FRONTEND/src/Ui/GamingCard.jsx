import HeroImageDes from "../assets/Hero-baseball-Image.png";

const GamingCard = () => {
    return (
        <div>
            {/* Gaming Card Description */}
            <h3 className="text-base md:text-lg border-4 font-medium mt-5 p-2 rounded-lg">
                Gaming <span className="text-amber-500">Card</span> Description 🥷
            </h3>
            {/* <GamingCard /> */}
            <aside className=" w-full mt-3  lg:w-[500px] h-[200px] shadow overflow-x-scroll scrollbar-none">
                <div className="h-full flex items-center justify-left">
                    <div className="flex gap-2 ">
                        <div className="flex-shrink-0 w-[120px]">
                            <img src={HeroImageDes} alt="Image 1" className="w-full h-full object-cover" />
                        </div>
                        <div className="flex-shrink-0 w-[120px]">
                            <img src={HeroImageDes} alt="Image 1" className="w-full h-full object-cover" />
                        </div>
                        <div className="flex-shrink-0 w-[120px]">
                            <img src={HeroImageDes} alt="Image 1" className="w-full h-full object-cover" />
                        </div>
                        <div className="flex-shrink-0 w-[120px]">
                            <img src={HeroImageDes} alt="Image 1" className="w-full h-full object-cover" />
                        </div>
                        <div className="flex-shrink-0 w-[120px]">
                            <img src={HeroImageDes} alt="Image 1" className="w-full h-full object-cover" />
                        </div>


                    </div>
                </div>
            </aside>
        </div>


    );
};

export default GamingCard;
