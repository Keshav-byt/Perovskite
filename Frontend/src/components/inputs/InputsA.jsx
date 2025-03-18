import data from '../../utils/columnHeadings.json';
const inputsA =()=>{
    return (
        <div className="p-4 m-4 overflow-y-scroll scrollbar-none border-r-2 border-gray-500">
            {data?.functional_groups[0]?.A &&
                Object.entries(data.functional_groups[0].A).map((value,key) => (
                    <div key={key} className={'mb-2'}>
                        <p>{value}</p>
                        <input placeholder="Enter Input" className={'border-1 p-1'}/>
                    </div>
                ))
            }
        </div>
    )
}
export default inputsA;