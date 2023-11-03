import { useEffect, useContext, useState } from 'react';
import HomeContext from '@/pages/api/home/home.context';

const libImages: { [key: string]: string } = {
  'squidpy': '/apps/squidpy.jpg',
  'pyteomics': '/apps/pyteomics.jpg',
  'qiime2': '/apps/qiime2.jpg',
  'scikitbio': '/apps/scikitbio.jpg',
  'scanpy': '/apps/scanpy.jpg',
  'custom': '/apps/customize.jpg',
};

export const LibCardSelect = () => {
  const {
    state: { selectedConversation, methods },
    handleUpdateConversation,
    dispatch: homeDispatch,
  } = useContext(HomeContext);
  
  const [selectedLib, setSelectedLib] = useState<string | undefined>(selectedConversation?.Lib);
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [customLibName, setCustomLibName] = useState('');
  const [customGitHubURL, setCustomGitHubURL] = useState('');
  const [customReadTheDocsURL, setCustomReadTheDocsURL] = useState('');
  const [customAPIHTML, setcustomAPIHTML]  = useState('');
  const [customLIBALIAS, setcustomLIBALIAS]  = useState('');

  useEffect(() => {
    const newLibs = [
      { id: 'squidpy', name: 'squidpy' },
      { id: 'pyteomics', name: 'pyteomics' },
      { id: 'qiime2', name: 'qiime2' },
      { id: 'scikitbio', name: 'scikitBio' },
      { id: 'scanpy', name: 'scanpy' },
      { id: 'custom', name: 'custom' },
    ];
  
    const existingLibIds = methods.map(lib => lib.id);
    const uniqueNewLibs = newLibs.filter(lib => !existingLibIds.includes(lib.id));
  
    if (uniqueNewLibs.length > 0) {
      // homeDispatch({ field: 'methods', value: [...methods, ...uniqueNewLibs] });
      homeDispatch({ field: 'methods', value: newLibs });
    }
  }, [homeDispatch]);
  
  const handleCardClick = (lib: { id: string, name: string }) => {
    setSelectedLib(lib.id);
    setShowCustomInput(lib.id === 'custom');
    if (lib.id !== 'custom' && selectedConversation) {
      handleUpdateConversation(selectedConversation, {
        key: 'Lib',
        value: lib.id,
      });
    }
  };
  
  useEffect(() => {
    if (selectedLib === 'custom' && selectedConversation) {
      handleUpdateConversation(selectedConversation, {
        key: 'Lib',
        value: customLibName,
      });
      handleUpdateConversation(selectedConversation, {
        key: 'new_lib_github_url',
        value: customGitHubURL,
      });
      handleUpdateConversation(selectedConversation, {
        key: 'new_lib_doc_url',
        value: customReadTheDocsURL,
      });
      handleUpdateConversation(selectedConversation, {
        key: 'api_html',
        value: customAPIHTML,
      });
      handleUpdateConversation(selectedConversation, {
        key: 'lib_alias',
        value: customLIBALIAS,
      });
    }
  }, [customLibName, customGitHubURL, customReadTheDocsURL, customAPIHTML, customLIBALIAS, selectedLib, selectedConversation, handleUpdateConversation]);
  return (
    <>
      <div className="select-container">
        <span className="text-[12px] text-black/50 dark:text-white/50 text-sm">
        Select the Lib
        </span>
      </div>
      <div className="card-select-container">
        {methods.map((lib) => (
          <div
            key={lib.id}
            className={`card ${selectedLib === lib.id ? 'card-selected' : ''}`}
            onClick={() => handleCardClick(lib)}
          >
            <img src={libImages[lib.id]} alt={lib.name} />
            <div>{lib.name}</div>
          </div>
        ))}
      </div>
      {showCustomInput && (
        <div className="custom-input-container">
          <div>
            <label>GitHub URL: </label>
            <input 
              type="text" 
              placeholder="GitHub URL" 
              style={{ color: 'black' }} 
              onChange={(e) => setCustomGitHubURL(e.target.value)}  // 添加这一行
            />
          </div>
          <div>
            <label>ReadTheDocs URL: </label>
            <input 
              type="text" 
              placeholder="ReadTheDocs URL" 
              style={{ color: 'black' }} 
              onChange={(e) => setCustomReadTheDocsURL(e.target.value)}  // 添加这一行
            />
          </div>
          <div>
            <label>Library Name: </label>
            <input 
              type="text" 
              placeholder="Library Name" 
              style={{ color: 'black' }} 
              onChange={(e) => setCustomLibName(e.target.value)} 
            />
          </div>
          <div>
            <label>API HTML: </label>
            <input 
              type="text" 
              placeholder="API HTML" 
              style={{ color: 'black' }} 
              onChange={(e) => setcustomAPIHTML(e.target.value)} 
            />
          </div>
          <div>
            <label>LIB ALIAS: </label>
            <input 
              type="text" 
              placeholder="LIB ALIAS (e.g. sklearn for scikit-learn)" 
              style={{ color: 'black' }} 
              onChange={(e) => setcustomLIBALIAS(e.target.value)} 
            />
          </div>
        </div>
      )}

      <style jsx>{`
        .select-container {
          display: flex;
          justify-content: flex-start;
          align-items: center;
          margin-bottom: 10px;
          font-family: 'Your Font Name', sans-serif; /* Replace 'Your Font Name' with the actual font name */
          font-size: 16px; /* Adjust font size as needed */
          font-weight: bold;
        }
        .card-select-container {
          display: flex;
          flex-wrap: wrap;
          align-items: flex-start;
          max-width: 2000px;
          width: 600px;
          justify-content: flex-start;
        }
        .card {
          cursor: pointer;
          width: 160px;
          height: 150px;
          padding: 10px;
          border: 1px solid gray;
          margin: 10px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          position: relative;
        }
        .card-selected {
          background-color: lightblue;
          color: #333333;
        }
        .card div {
          position: absolute;
          bottom: 0;
          left: 0;
          width: 100%;
          padding: 5px;
          background-color: rgba(255, 255, 255, 0.8);
        }
        .card img {
          max-width: 100%;
          max-height: calc(100% - 5px); 
          margin-bottom: -20px;
        }
        .custom-input-container div {
          display: flex;
          align-items: center;
          margin-bottom: 20px;
        }
        label {
          font-weight: bold;
          margin-right: 20px;
        }
      `}</style>
    </>
  );
};