import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom';
import { navigation } from '../contsants';
import { FaMicrophone, FaUserCircle } from 'react-icons/fa';
import { disablePageScroll, enablePageScroll } from "scroll-lock";
import Logo from "../assets/logo1.png"
import MenuSvg from "../assets/svg/MenuSvg";
import { HamburgerMenu } from "./design/Header";
import Button from './Button';

const Header = () => {
    const pathname = useLocation();
    const [openNavigation, setOpenNavigation] = useState(false);
    const toggleNavication = () => {
        if (openNavigation) {
            setOpenNavigation(false);
            enablePageScroll();
        } else {
            setOpenNavigation(true);
            disablePageScroll();
        }
    };

    const handleClick = () => {
        if (!openNavigation) return;

        enablePageScroll();
        setOpenNavigation(false);
    };
    return (
        <>
            <div className={`fixed top-0 left-0 w-full z-50 border-b border-n-6 lg:ng-n-8/90 lg:backdrop-blur-sm ${openNavigation ? "bg-n-8" : "bg-n-8/90 backdrop-blur-sm"
                }`}>
                <div className='flex items-center justify-between px-2  lg:px-7.5 '>
                    {/* <Link href="" className='block w-[12rem] x:mr-8 font-rubikVinyl'> Baseball League</Link> */}
                    <Link href=""> <img src={Logo} className='lg:w-24 w-20' alt="" /></Link>

                    <nav className={`${openNavigation ? "flex" : "hidden"
                        }  fixed top-[5rem] left-0
            right-0 bottom-0 bg-n-8 lg:static lg:flex
            lg:mx-auto lg:bg-transparent`}
                    >
                        <div className='relative 2-2 flex flex-col
            items-center justify-center m-auto lg:flex-row'>
                            {navigation.map((item) => (
                                <a
                                    key={item.id}
                                    href={item.url}
                                    onClick={handleClick}
                                    className={`block relative font-code text-2xl
                uppercase text-n-1 translate-colors hover:text-color-1 ${item.onlyMobile ? "lg:hidden" : ""
                                        } px-6 py-6 md:py-8 lg:-mr-0.25 lg:text-xs lg:font-semibold  ${item.url === pathname.hash
                                            ? "z-2 lg:text-n-1"
                                            : "lg:text-n-1/50"
                                        } lg:leading-5 lg:hover:text-n-1 xl:px-12`}
                                >
                                    {item.title}
                                </a>
                            ))}
                        </div>
                        <HamburgerMenu />
                    </nav>

                    <div className="flex items-center gap-4 ">
                        <Button
                            className="ml-auto lg:hidden"
                            px="px-3"
                            onClick={toggleNavication}

                        >
                            <MenuSvg openNavigation={openNavigation} />
                        </Button>

                    </div>

                </div>

            </div>
        </>
    )
}

export default Header