package main;

import com.twitter.chill.config.Config;
import org.apache.log4j.Logger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Properties;

/**
 * Created by hadoop on 17-4-5.
 */
public class Configuration {

    public Properties properties;

    private Logger logger;

    private Configuration(){
        properties = new Properties();

        logger = Logger.getLogger(Configuration.class.getName());

        try {
            properties.load(new InputStreamReader(
                   Configuration.class.getClassLoader().getResourceAsStream("conf.properties"), "UTF-8"));
        } catch (FileNotFoundException e) {
            logger.warn("Properties File Not found");
        } catch (IOException e) {
            logger.warn("Properties File Read Error");
        }
    }

    private static Configuration config;

    public static Configuration getConfig() {
        if (config == null) {
            synchronized (Configuration.class){
                config = new Configuration();
            }
        }
        return config;
    }
}
