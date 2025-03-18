const ResultPage = () => {
    

    return (
        <div className="w-[75%] m-auto text-white overflow-y-auto scrollbar-none">
            <div className="backdrop-blur-xl h-[40vh] mt-10 flex flex-col justify-center items-center text-6xl rounded-2xl result-box-border">
            {/*    Funtional Group*/}
                <h1>FUNTIONAL GROUP</h1>
            </div>
            <div className={'flex  flex-wrap justify-between '}>
                <div className="backdrop-blur-xl h-[30vh] mt-5 w-[48%] rounded-2xl result-box-border ">
                    <div>
                        <h1 className={'text-2xl p-2'}>The given functional group is a :</h1>
                    </div>

                    <div className={'flex flex-col justify-center items-center relative h-[75%] '}>
                        <h1 className={'text-5xl '}>Conductor</h1>
                    </div>
                </div>

                <div className="backdrop-blur-xl h-[30vh] mt-5 w-[48%] rounded-2xl result-box-border ">
                    {/*    bandgap*/}
                    <div>
                        <h1 className={'text-2xl p-2'}>Band Gap of the given Functional Group:</h1>
                    </div>

                    <div className={'flex flex-col justify-center items-center relative h-[75%] '}>
                        <h1 className={'text-5xl '}>Null</h1>
                    </div>

                </div>
            </div>
            <div>
                {/*    info about energy gap*/}
                <div className="backdrop-blur-xl h-[30vh] mt-5 rounded-2xl result-box-border p-2">
                    <h1 className={'text-2xl ml-2'}>What is BandGap?</h1>
                    <h1 className={'text-lg mt-2 ml-2'}>Lorem ipsum and info about band gaps</h1>

                </div>
            </div>
        </div>
    )
}
export default ResultPage